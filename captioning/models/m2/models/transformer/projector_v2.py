import pdb

import torch
from torch import nn
# from models.transformer.utils import sinusoid_encoding_table

# 返回二维矩阵，用于
def sinusoid_encoding_table(seq_len, d_model):
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(d_model // 2, dtype=torch.float32).view(1, -1)
    sin = torch.sin(pos / 10000 ** (2 * dim / d_model))
    cos = torch.cos(pos / 10000 ** (2 * dim / d_model))

    out = torch.zeros((pos.shape[0], d_model))
    out[:, ::2] = sin
    out[:, 1::2] = cos
    return out

# 将O和T进行处理拼接
class Projector(nn.Module):
    def __init__(self, f_obj, f_grid, f_tag, f_out, drop_rate=0.3):
        super(Projector, self).__init__()

        # for objects O
        self.obj_mlp1 = nn.Sequential(
            nn.LayerNorm(f_obj), nn.Linear(f_obj, f_out), nn.Dropout(p=drop_rate)
        )

        # for grids
        self.grids_keys = ("whole", "nine", "sixteen", "tweentyfive")
        for k in self.grids_keys:
            mlp1 = nn.Sequential(
                nn.LayerNorm(f_grid),
                nn.Linear(f_grid, f_out),
                nn.Dropout(p=drop_rate)
            )
            setattr(self, f"grid_mlp_{k}", mlp1)

            if k == 'whole':
                num_embeddings = 1
            elif k == 'nine':
                num_embeddings = 9
            elif k == 'sixteen':
                num_embeddings = 16
            elif k == 'tweentyfive':
                num_embeddings = 25
            else:
                raise KeyError

            pos = nn.Embedding.from_pretrained(
                sinusoid_encoding_table(num_embeddings, f_out), freeze=True
            )
            setattr(self, f"grid_pos_{k}", pos)

            input_index = torch.arange(num_embeddings)
            setattr(self, f"grid_index_{k}", input_index)


        # for tags*
        self.tag_mlp1 = nn.Sequential(
            nn.LayerNorm(f_tag), nn.Linear(f_tag, f_out), nn.Dropout(p=drop_rate)
        )


    def forward(self, obj, grid, tag):
        embed = []

        # object O
        obj_mask = (torch.sum(torch.abs(obj), dim=-1) == 0) # N x S # N个obj，最后一维大小S
        obj_embed = self.obj_mlp1(obj)  # -> (b_s, 50, f_out)
        obj_embed[obj_mask] = 0.    # 通过obj_mask来选择obj_embed中对应位置的向量赋值为0，防止无效的obj分量产生干扰
        embed.append(obj_embed)

        # grids
        for k in self.grids_keys:
            # pos_k = grid[k]["pos"]
            embed_k = grid[k]
            mlp_1 = getattr(self, f"grid_mlp_{k}")  # -> (b_s, 9, f_out)
            mlp_pos = getattr(self, f"grid_pos_{k}")
            grid_index = getattr(self, f"grid_index_{k}")
            grid_index = grid_index.to(embed_k.device)

            embed_k = mlp_1(embed_k) + mlp_pos(grid_index) # grid向量 + 位置编码
            embed.append(embed_k)


        # tags
        tag_mask = (torch.sum(torch.abs(tag), dim=-1) == 0) # N x S
        tag_embed = self.tag_mlp1(tag)  # -> (b_s, 50, f_out)
        tag_embed[tag_mask] = 0.
        embed.append(tag_embed)

        # tags
        for k in self.tags_keys:
            embed_k = tag[k]["embed"]
            mlp_2 = getattr(self, f"tag_mlp_{k}")
            embed_k = mlp_2(embed_k)  # -> (b_s, 9*k, f_out)
            embed.append(embed_k)

        return torch.cat(embed, dim=1)  # 将obj、grid、tag进行拼接[o, g..., t...]
        # -> (b_s, 50+1+9+16+25+k, f_out)