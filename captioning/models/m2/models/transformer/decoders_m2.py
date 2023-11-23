import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from .attention import MultiHeadAttention
from .utils import sinusoid_encoding_table, PositionWiseFeedForward
from ..containers import Module, ModuleList


class MeshedDecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True, # self的att是可记录的
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False, # enc的att是不可被记录的
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout) # 进行位置编码

        self.fc_alpha1 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha2 = nn.Linear(d_model + d_model, d_model)
        self.fc_alpha3 = nn.Linear(d_model + d_model, d_model)

        self.init_weights()

    # 初始化全连接层权重
    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_alpha1.weight)
        nn.init.xavier_uniform_(self.fc_alpha2.weight)
        nn.init.xavier_uniform_(self.fc_alpha3.weight)
        nn.init.constant_(self.fc_alpha1.bias, 0)
        nn.init.constant_(self.fc_alpha2.bias, 0)
        nn.init.constant_(self.fc_alpha3.bias, 0)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)    # 输入和mask，获取att
        self_att = self_att * mask_pad  # 进行掩膜

        enc_att1 = self.enc_att(self_att, enc_output[:, 0], enc_output[:, 0], mask_enc_att) * mask_pad  # 以计算的att为q，enc的输出为k和v，进行att的掩膜和填充计算
        enc_att2 = self.enc_att(self_att, enc_output[:, 1], enc_output[:, 1], mask_enc_att) * mask_pad
        enc_att3 = self.enc_att(self_att, enc_output[:, 2], enc_output[:, 2], mask_enc_att) * mask_pad

        alpha1 = torch.sigmoid(self.fc_alpha1(torch.cat([self_att, enc_att1], -1))) # 分别计算后，与att作cat，后通过全连接层，->sigmoid激活，得到权重参数
        alpha2 = torch.sigmoid(self.fc_alpha2(torch.cat([self_att, enc_att2], -1)))
        alpha3 = torch.sigmoid(self.fc_alpha3(torch.cat([self_att, enc_att3], -1)))

        enc_att = (enc_att1 * alpha1 + enc_att2 * alpha2 + enc_att3 * alpha3) / np.sqrt(3)  # 将权重参数*enc的输出进行加权，并除以根号3
        enc_att = enc_att * mask_pad    # 进行掩膜

        ff = self.pwff(enc_att) # 对该att进行位置编码
        ff = ff * mask_pad      # 进行填充
        return ff


class MeshedDecoderM2(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(MeshedDecoderM2, self).__init__()
        self.d_model = d_model  # model的维数
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)  # word嵌入随机初始化
        self.pos_emb = nn.Embedding.from_pretrained(sinusoid_encoding_table(max_len + 1, d_model, 0), freeze=True)  # 使用已预训练好的词向量初始化，此处为位置编码
        self.layers = ModuleList(
            [MeshedDecoderLayer(d_model, d_k, d_v, h, d_ff, dropout, self_att_module=self_att_module,
                                enc_att_module=enc_att_module, self_att_module_kwargs=self_att_module_kwargs,
                                enc_att_module_kwargs=enc_att_module_kwargs) for _ in range(N_dec)])    # 建立N层dec
        self.fc = nn.Linear(d_model, vocab_size, bias=False)    # 全连接层
        self.max_len = max_len          # 最大长度
        self.padding_idx = padding_idx  # 填充索引
        self.N = N_dec                  # N层dec

        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())   # 记录状态
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, input, encoder_output, mask_encoder):
        # input (b_s, seq_len)
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (b_s, seq_len, 1)   非填充处mask，此处表示存在
        mask_self_attention = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device),    # 返回上三角全为1的矩阵，其尺寸为seq_len
                                         diagonal=1)
        mask_self_attention = mask_self_attention.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len) # 进行拓维
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()    # 在上三角矩阵的基础上，若该位置为填充符，元素+1
        mask_self_attention = mask_self_attention.gt(0)  # (b_s, 1, seq_len, seq_len)   # 严格大于0的为1，小于等于0的则置为0
        if self._is_stateful:
            self.running_mask_self_attention = torch.cat([self.running_mask_self_attention, mask_self_attention], -1)   # 将运行中的mask_att和mask_att进行cat
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (b_s, seq_len)   从1到seq_len+1扩充数组，并扩展乘b_s条
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0) # mask_q缩减-1维，若其为0，则值置为0
        if self._is_stateful:
            self.running_seq.add_(1)    # in_place的相加x+y操作，会将结果存储到原来的x中
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)  # 对input进行编码，对位置进行编码，并将其相加
        for i, l in enumerate(self.layers):
            out = l(out, encoder_output, mask_queries, mask_self_attention, mask_encoder)   # 将嵌入依次输入dec，获得输出

        out = self.fc(out)  # 全连接层
        return F.log_softmax(out, dim=-1)   # 返回log_softmax分数
