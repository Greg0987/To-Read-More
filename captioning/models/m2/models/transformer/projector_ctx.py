import torch
from torch import nn

def sinusoid_encoding_table(seq_len, d_model, padding_idx=None):
    pos = torch.arange(seq_len, dtype=torch.float32).view(-1, 1)
    dim = torch.arange(d_model // 2, dtype=torch.float32).view(1, -1)
    sin = torch.sin(pos / 10000 ** (2 * dim / d_model))
    cos = torch.cos(pos / 10000 ** (2 * dim / d_model))

    out = torch.zeros((pos.shape[0], d_model), device=pos.device)
    out[:, ::2] = sin
    out[:, 1::2] = cos
    if padding_idx is not None:
        out[padding_idx] = 0
    return out

# 映射类，将O和T进行处理拼接
class Projector_ctx(nn.Module):
    def __init__(self, f_obj, f_grid, f_tag, f_out, drop_rate=0.3):
        super().__init__()

        # for objects O
        self.obj_mlp1 = nn.Sequential(  # 全连接层1，包含归一化层、线性连接层、dropout层
            nn.LayerNorm(f_obj), nn.Linear(f_obj, f_out), nn.Dropout(p=drop_rate)
        )

        self.obj_mlp2 = nn.Sequential(
            nn.LayerNorm(f_grid), nn.Linear(f_grid, f_out), nn.Dropout(p=drop_rate)
        )

        self.keys = ("whole", "five", "nine")

        # for grids
        for k in self.keys:
            mlp1 = nn.Sequential(   # 全连接层1
                nn.LayerNorm(f_tag), nn.Linear(f_tag, f_out), nn.Dropout(p=drop_rate)
            )
            mlp2 = nn.Sequential(
                nn.LayerNorm(f_grid), nn.Linear(f_grid, f_out), nn.Dropout(p=drop_rate)
            )
            setattr(self, f"txt_mlp1_{k}", mlp1)
            setattr(self, f"txt_mlp2_{k}", mlp2)

            if k == "whole":
                num_embeddings = 1
            elif k == "five":
                num_embeddings = 5
            elif k == "nine":
                num_embeddings = 9
            else:
                raise KeyError

            pos = nn.Embedding.from_pretrained(
                sinusoid_encoding_table(num_embeddings, f_out), freeze=True
            )  # 位置编码信息
            setattr(self, f"txt_pos_{k}", pos)

    def forward(self, obj, grid, tag):
        img = grid[:, None, :]   # (b_s, _, 768)
        embed = []

        # object O
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0)  # N x S
        obj_embed = self.obj_mlp1(obj) + self.obj_mlp2(img) # 直接相加：o = O + I    # -> (b_s, 1, f_out)
        obj_embed[obj_mask] = 0.
        embed.append(obj_embed)

        # ctx T
        for k in self.keys:
            pos_k = tag[k]["pos"]
            embed_k = tag[k]["embed"]   # (b_s, 9*k, d)
            mlp1 = getattr(self, f"txt_mlp1_{k}")   # -> (b_s, 9*k, f_out)
            mlp2 = getattr(self, f"txt_mlp2_{k}")   # -> (b_s, 1, f_out)
            mlp_pos = getattr(self, f"txt_pos_{k}")
            embed_k = mlp1(embed_k) + mlp2(img) + mlp_pos(pos_k)    # 直接相加：t1 = T1 + I + 位置编码；t5 = T2 + I +位置编码...
            # 广播机制 -> (b_s, 9*k, f_out)
            embed.append(embed_k)

        return torch.cat(embed, dim=1)  # 将O和T的嵌入进行concat：[o, t1, t5, t9]
        # 在第1维上进行concat -> (b_s, 1+1*k+5*k+9*k, f_out)