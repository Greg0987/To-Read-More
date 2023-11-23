from torch.nn import functional as F
from .utils import PositionWiseFeedForward
import torch
from torch import nn
from .attention import MultiHeadAttention

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering  # id匹配重新排序标记
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)    # 多头注意力类
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)    # 位置编码类

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        att = self.mhatt(queries, keys, values, attention_mask, attention_weights)  # 返回多头注意力att
        ff = self.pwff(att) # 将att输入并返回位置编码
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model  # model的维度
        self.dropout = dropout  # dropout概率
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])    # n层enc
        self.padding_idx = padding_idx  # 填充的idx

    def forward(self, input, attention_weights=None):
        # input (b_s, seq_len, d_in)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)   # 对attention进行掩膜

        outs = []
        out = input
        for l in self.layers:   # 每一层enc
            out = l(out, out, out, attention_mask, attention_weights)   # 将out，mask，以及weights输入各层enc
            outs.append(out.unsqueeze(1))   # 展开一个维度，并加入outs

        outs = torch.cat(outs, 1)   # 将所有out进行拼接
        return outs, attention_mask


class MemoryAugmentedEncoderM2(MultiLevelEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super(MemoryAugmentedEncoderM2, self).__init__(N, padding_idx, **kwargs)
        self.fc = nn.Linear(d_in, self.d_model)         # 全连接层
        self.dropout = nn.Dropout(p=self.dropout)       # dropout层
        self.layer_norm = nn.LayerNorm(self.d_model)    # 归一化

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))    # 输入-> fc -> relu激活
        out = self.dropout(out)         # dropout
        out = self.layer_norm(out)      # 归一化
        return super(MemoryAugmentedEncoderM2, self).forward(out, attention_weights=attention_weights)    # 前向传播
