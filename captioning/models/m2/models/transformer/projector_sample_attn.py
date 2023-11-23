import torch
from torch import nn
import numpy as np

class SampleAttentionLayer(nn.Module):
    def __init__(self, d_model, d_k, d_v, q_before, q_after, d_reduction=None):
        """
        :param d_model: 模型维度, 默认512
        :param d_k:     输入的k
        :param d_v:     输入的v
        :param q_before:     采样前的q
        :param q_after:     采样后的q
        :param d_reduction: 降维方法
        """
        super(SampleAttentionLayer, self).__init__()

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.d_reduction = d_reduction
        self.q_before = q_before
        self.q_after = q_after
        if self.q_before == self.q_after:
            self.d_reduction = None

        # 用于fc降维
        self.fc_d_q = nn.Linear(q_before, q_after)
        # 用于sample降维
        self.sample_embedding = nn.Embedding(q_after, d_model)
        self.sample_seq = torch.from_numpy(np.arange(q_after, dtype=np.int64))
        # 用于conv降维
        self.conv = nn.Conv1d(d_model, d_model, q_before//q_after, q_before//q_after, groups=1)


        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.xavier_uniform_(self.fc_d_q.weight)

        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)
        nn.init.constant_(self.fc_d_q.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q_ = queries.view(b_s, nq, -1).permute(0, 2, 1)  # (b_s, nq, d_model) -> (b_s, d_model, nq)
        # 对q进行降维
        if self.d_reduction == 'fc':
            q_ = self.fc_d_q(q_).view(b_s, self.d_model, self.q_after).permute(0, 2, 1)  # (b_s, d_model, q_after) -> (b_s, q_after, d_model)
        elif self.d_reduction == 'sample':
            sample_emb = self.sample_embedding(self.sample_seq.to(queries.device)).unsqueeze(0)  # (1, q_after, d_model)
            scores = torch.matmul(sample_emb, q_) / np.sqrt(self.d_model)  # (b_s, q_after, nq)
            scores = torch.softmax(scores.squeeze(), dim=-1)                 # (b_s, q_after, nq)
            sample = torch.multinomial(scores.view(b_s * self.q_after, self.q_before), 1, replacement=True).squeeze() # (q_after)
            q_ = queries.reshape(b_s * nq, self.d_model).index_select(0, sample)  # (b_s * q_after, d_model
            q_ = q_.view(b_s, self.q_after, self.d_model)       # (b_s, q_after, d_model)
        elif self.d_reduction == 'conv':
            q_ = self.conv(q_).transpose(1, 2)  # (b_s, q_after, d_model)
        else:
            q_ = queries

        q = self.fc_q(q_).view(b_s, self.q_after, self.d_k)  # (b_s, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.d_k).permute(0, 2, 1)  # (b_s, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.d_v)   # (b_s, nk, d_v)

        att = torch.matmul(q, k) / (self.d_k ** 0.5)  # (b_s, q_after, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        att = torch.softmax(att, dim=-1)  # (b_s, q_after, nk)
        out = torch.matmul(att, v).contiguous()  # (b_s, q_after, d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

# 包一下attention作layernorm
class SampleAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, q_before, q_after, d_reduction=None, dropout=0.3):
        super(SampleAttention, self).__init__()
        self.attention = SampleAttentionLayer(d_model, d_k, d_v, q_before, q_after, d_reduction)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_reduction = d_reduction

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        out = self.attention(queries, keys, values, attention_mask, attention_weights)
        out = self.dropout(out)
        if self.d_reduction:
            out = self.layer_norm(queries)
        else:
            out = self.layer_norm(queries + out)
        return out

# 映射类，将O和T进行处理拼接
class Projector_attn(nn.Module):
    def __init__(self, f_obj, f_grid, f_tag, f_out, topk, d_reduction=None, drop_rate=0.3):
        super().__init__()

        # for objects O
        self.obj_mlp1 = nn.Sequential(  # 全连接层1，包含归一化层、线性连接层、dropout层
            nn.LayerNorm(f_obj), nn.Linear(f_obj, f_out-1), nn.Dropout(p=drop_rate)
        )

        self.keys = ("whole", "four", "nine", "sixteen", "twentyfive")   # 5种剪裁的图片网格
        self.keys_ = (1, 2, 3, 4, 5) # 5种剪裁
        self.topk = topk

        # for grids
        for k in self.keys:
            mlp1 = nn.Sequential(   # 全连接层1
                nn.LayerNorm(f_grid), nn.Linear(f_grid, f_out-1), nn.Dropout(p=drop_rate)
            )
            setattr(self, f"grid_mlp_{k}", mlp1)

            if k == 'whole':
                num_embeddings = 1
            elif k =='four':
                num_embeddings = 4
            elif k == 'nine':
                num_embeddings = 9
            elif k == 'sixteen':
                num_embeddings = 16
            elif k == 'twentyfive':
                num_embeddings = 25
            else:
                raise KeyError

            # pos_embed = nn.Parameter(torch.zeros(1, num_embeddings, f_out-1))  # 位置编码 # (1, 1, 512)
            # pos_drop = nn.Dropout(p=drop_rate)

            # nn.init.trunc_normal_(pos_embed, std=0.02)
            # setattr(self, f"grid_pos_{k}", pos_embed)
            # setattr(self, f"grid_pos_drop_{k}", pos_drop)

        # for tags
        for k in self.keys:
            mlp2 = nn.Sequential(
                nn.LayerNorm(f_tag), nn.Linear(f_tag, f_out-1), nn.Dropout(p=drop_rate)
            )
            setattr(self, f"tag_mlp_{k}", mlp2)

            if k == 'whole':
                num_embeddings = 1
            elif k =='four':
                num_embeddings = 4
            elif k == 'nine':
                num_embeddings = 9
            elif k == 'sixteen':
                num_embeddings = 16
            elif k == 'twentyfive':
                num_embeddings = 25
            else:
                raise KeyError

            pos_embed = nn.Parameter(torch.zeros(1, num_embeddings * topk, f_out-1))  # 位置编码
            pos_drop = nn.Dropout(p=drop_rate)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            setattr(self, f"tag_pos_{k}", pos_embed)
            setattr(self, f"tag_pos_drop_{k}", pos_drop)
            cross_attn = SampleAttention(d_model=512, d_k=64, d_v=64,
                                        q_before = num_embeddings * topk,
                                        q_after = int(np.sqrt(num_embeddings) * topk),
                                        d_reduction=d_reduction, dropout=drop_rate)
            setattr(self, f"cross_attn_{k}", cross_attn)

        self.self_attn = SampleAttention(d_model=512, d_k=64, d_v=64,
                                         q_before=(1+4+9+16+25)*topk, q_after=(1+4+9+16+25)*topk,
                                         d_reduction=None, dropout=drop_rate)

        # 看vit是默认升到4倍维度
        self.ffn = nn.Sequential(
            nn.Linear(f_out, 4 * f_out), nn.ReLU(), nn.Linear(4 * f_out, f_out), nn.Dropout(p=drop_rate))
        self.layer_norm = nn.LayerNorm(f_out)

    def forward(self, obj, grid, tag):
        # img = vis_ctx[:, None, :]   # (b_s, _, 768)
        embed_obj = []
        embed_grid = []
        embed_tag = []
        embed_tag_attn = []

        # object O
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)  # N x S
        obj_embed = self.obj_mlp1(obj) # -> (b_s, 1, f_out)
        obj_embed[obj_mask] = 0.
        # 模态0
        x = 0
        obj_embed = torch.cat((x * torch.ones((obj_embed.shape[0],obj_embed.shape[1], 1)).cuda(), obj_embed), dim=-1)  # (b_s, 1+f_out)
        embed_obj.append(obj_embed)
        embed_obj_ = torch.cat(embed_obj, dim=1)  # (b_s, 50, 512)

        # grids G
        x = 1
        for k in self.keys:
            embed_k = grid[k]  # (b_s, 9*k, d)
            # pos_k = grid[k]["pos"]  # (b_s, 9*k, d)
            mlp1 = getattr(self, f"grid_mlp_{k}")  # -> (b_s, 9*k, f_out)
            pos_embed = getattr(self, f"grid_pos_{k}")
            pos_drop = getattr(self, f"grid_pos_drop_{k}")
            embed_k = pos_drop(mlp1(embed_k) + pos_embed)    # 加上位置编码
            # embed_k = mlp1(embed_k)
            embed_k = pos_drop(embed_k + pos_embed)
            embed_k = torch.cat((x * torch.ones((embed_k.shape[0],embed_k.shape[1], 1)).cuda(), embed_k), dim=-1)  # (b_s, 1+f_out) # 加上模态1
            embed_grid.append(embed_k)
        embed_grid_ = torch.cat(embed_grid, dim=1)  # (b_s, 1+4+9+16+25, 512)

        # tags T
        x = 2
        for k in self.keys:
            embed_k = tag[k]
            # pos_k = tag[k]["pos"]
            mlp2 = getattr(self, f"tag_mlp_{k}")
            # mlp_pos = getattr(self, f"tag_pos_{k}")
            pos_embed = getattr(self, f"tag_pos_{k}")
            pos_drop = getattr(self, f"tag_pos_drop_{k}")
            embed_k = pos_drop(mlp2(embed_k) + pos_embed)
            # embed_k = mlp2(embed_k)
            embed_k = pos_drop(embed_k + pos_embed)
            embed_k = torch.cat((x * torch.ones((embed_k.shape[0],embed_k.shape[1], 1)).cuda(), embed_k), dim=-1)  # (b_s, 1+f_out)
            embed_tag.append(embed_k)
        embed_tag_ = torch.cat(embed_tag, dim=1)  # (b_s, (1+4+9+16+25)*k, 512)

        # 标签内部作自注意
        embed_tag_ = self.self_attn(embed_tag_, embed_tag_, embed_tag_)
        embed_tag__ = embed_tag_.view(embed_tag_.shape[0], -1, self.topk, embed_tag_.shape[-1])
        # 标签与网格作交叉注意
        index = 0
        for i, j in zip(self.keys_, self.keys):
            embed_t = embed_tag__[:, index:index + i**2, :, :]    # (b_s, 1+4+9+16+25, topk, 512)
            embed_g = embed_grid_[:, index:index + i**2, :]  # (b_s, 1+4+9+16+25, 512)
            embed_t = embed_t.view(embed_t.shape[0], i**2 * self.topk, embed_t.shape[-1])  # (b_s, (1+4+9+16+25)*topk, 512)
            cross_attn = getattr(self, f"cross_attn_{j}")
            embed_t = cross_attn(embed_t, embed_g, embed_g) # 此处tag已经交叉注意降维，从i**2 -> i
            embed_tag_attn.append(embed_t)
            index += i**2
        embed_tag_attn = torch.cat(embed_tag_attn, dim=1)  # (b_s, (1+4+9+16+25)*topk, 512)
        # 前向反馈
        # embed_tag_ = self.ffn(embed_tag_attn) + embed_tag_attn
        embed_tag_ = self.layer_norm(self.ffn(embed_tag_attn) + embed_tag_attn)  # (b_s, (1+4+9+16+25)*topk, 512)

        return torch.cat((embed_obj_, embed_grid_, embed_tag_), dim=1)
        # 将obj、grid、tag进行拼接[o, g..., t...]
        # -> (b_s, 50+（1+9+16+25）*k , 512)
