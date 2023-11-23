"""
Instruction to use meshed_memory_transformer (https://arxiv.org/abs/1912.08226)

pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git

Note:
Currently m2transformer is not performing as well as original transformer. Not sure why? Still investigating.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import utils
import numpy as np
import pdb

from .TransformerModel import subsequent_mask

import os
import sys
sys.path.append(os.path.dirname(__file__))

from .m2.models.transformer import Transformer, MemoryAugmentedEncoder, MeshedDecoder, ScaledDotProductAttentionMemory, Projector_ctx

from .TransformerModel import TransformerModel

class CTXM2TransformerModel(TransformerModel):

    def __init__(self, opt):
        super(CTXM2TransformerModel, self).__init__(opt)
        delattr(self, 'att_embed')
        self.opt = opt
        self.k = opt.topk
        self.att_embed = lambda x: x  # The visual embed is in the MAEncoder
        # Notes: The dropout in MAEncoder is different from my att_embed, mine is 0.5?
        # Also the attention mask seems wrong in MAEncoder too...intersting
        delattr(self, 'embed')
        self.embed = lambda x: x

    def make_model(self, src_vocab, tgt_vocab, N_enc=3, N_dec=3,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        # 编码器，N层attention layer，用0填充，使用指定的SDPAttention
        encoder = MemoryAugmentedEncoder(N_enc, 0, d_in=512, attention_module=ScaledDotProductAttentionMemory,
                                         attention_module_kwargs={'m':40})
        # 解码器，输出词表概率，句子最大长度54，N层layer，用'-1'填充
        decoder = MeshedDecoder(tgt_vocab, 54, N_dec, -1)  # -1 is padding;
        # 多模态映射器，将obj、grid、tag进行拼接
        projector = Projector_ctx(
            f_obj=2054, f_grid=768, f_tag=512, f_out=d_model, drop_rate=dropout
        )
        model = Transformer(0, encoder, decoder, projector) # 0 is bos
        return model

    # 返回原本的logit输出
    def logit(self, x):  # unsafe way
        return x  # M2transformer always output logsoftmax

    # 准备输入特征，获取注意力及其掩码、序列及其掩码
    # 默认输出att_feats, seq=None, att_masks=1, seq_mask=None
    def _prepare_feature_forward(self, att_feats, att_masks=None, seq=None):
        # att_feats, att_masks = self.clip_att(att_feats, att_masks)

        # 注意力掩码
        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)   # 全1矩阵
        att_masks = att_masks.unsqueeze(-2) # b x 1 x att_len   # 增加一个维度

        if seq is not None:
            # crop the last one
            # seq = seq[:,:-1]
            # 序列掩码，不包括eos和pad
            seq_mask = (seq.data != self.eos_idx) & (seq.data != self.pad_idx)
            # 第一个位置为1，即bos
            seq_mask[:,0] = 1 # bos

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)

            seq_per_img = seq.shape[0] // att_feats.shape[0]
            # 生成多个句子，则复制特征和掩码
            if seq_per_img > 1:
                att_feats, att_masks = utils.repeat_tensors(seq_per_img,
                    [att_feats, att_masks]
                )
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask

    # 准备输入特征
    # 默认输出fc_feats=0, att_feats[...,:0], memory, att_masks
    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        """

        :param fc_feats: 为0
        :param att_feats: 注意力特征
        :param att_masks: None
        :return:
        """
        # att_feats, seq=None, att_masks=1, seq_mask=None
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        # 返回编码器编码，注意力掩码
        memory, att_masks = self.model.encoder(att_feats)

        return fc_feats[..., :0], att_feats[..., :0], memory, att_masks

    # 进行SCST训练
    def _sample(self, obj_f, grid_f, tag_f, opt={}):

        sample_method = opt.get('sample_method', 'greedy')
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        sample_n = int(opt.get('sample_n', 1))
        group_size = opt.get('group_size', 1)
        output_logsoftmax = opt.get('output_logsoftmax', 1)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        remove_bad_endings = opt.get('remove_bad_endings', 0)

        # print('+' * 20)                             # sample
        # print('sample_method: ', sample_method)     # greedy
        # print('beam_size: ', beam_size)             # 5
        # print('sample_n: ', sample_n)               # 1
        # print('group_size: ', group_size)           # 1
        # print('+' * 20)

        att_feats = self.model.projector(obj_f, grid_f, tag_f)
        att_masks = None    # set att_masks to None if attention features have the same length
        # 使用M2transformer，fc_feats为0
        fc_feats = torch.empty((att_feats.shape[0], 0)).to(att_feats.device)    # fc向量置为空

        # 要么进行贪婪搜索/束搜索
        if beam_size > 1 and sample_method in ['greedy', 'beam_search']:    # 束搜索
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)   # 输入0，特征，None
        # 要么进行多样性束搜索
        if group_size > 1:
            return self._diverse_sample(fc_feats, att_feats, att_masks, opt)

        # 要么进行普通采样
        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size * sample_n)

        # 此处p_fc=0, p_att是特征的第一个, pp_att是经过encoder编码的memory, p_att是经过encoder编码的注意力特征
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        if sample_n > 1:
            p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(sample_n,
                                                                                      [p_fc_feats, p_att_feats,
                                                                                       pp_att_feats, p_att_masks]
                                                                                      )
        # 用于存储bs个字典（即一张图片一个字典），代表一个样本序列，用于记录生成序列的三元组
        trigrams = []  # will be a list of batch_size dictionaries

        seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        for t in range(self.seq_length + 1):
            if t == 0:  # input <bos>
                it = fc_feats.new_full([batch_size * sample_n], self.bos_idx, dtype=torch.long) # 第0步it为bos
            # 输入起始张量和相关特征，获得下一个词的所有概率值和状态【比如获得了词A，词B，词C】
            # 输入bos，0，特征第一个，memory，注意力掩码，state[]
            # 传到decoder，输出下一个词的概率，和state
            logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state,
                                                      output_logsoftmax=output_logsoftmax)
            # 解码约束，避免生成重复词语
            if decoding_constraint and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                # 将上一个时间步生成的词，作为索引，填充负无穷大的值
                tmp.scatter_(1, seq[:, t - 1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp

            # 移除不良结尾
            if remove_bad_endings and t > 0:
                tmp = logprobs.new_zeros(logprobs.size())
                # 如果上一个时间步生成的词是不良结尾词，则将其概率值置为负无穷大
                prev_bad = np.isin(seq[:, t - 1].data.cpu().numpy(), self.bad_endings_ix)
                # Make it impossible to generate bad_endings
                tmp[torch.from_numpy(prev_bad.astype('uint8')), 0] = float('-inf')
                logprobs = logprobs + tmp

            # Mess with trigrams
            # Copy from https://github.com/lukemelas/image-paragraph-captioning
            if block_trigrams and t >= 3:   # 判断是否需要处理三元组
                # Store trigram generated at last step
                # 存储上一步生成的三元组，获取前两个词的索引
                prev_two_batch = seq[:, t - 3:t - 1]
                # 遍历每个样本
                for i in range(batch_size):  # = seq.size(0)
                    # 把前两个词pre_2作为键，当前（第3个词）作为值，存储在列表对应的字典中
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current = seq[i][t - 1]
                    if t == 3:  # initialize
                        trigrams.append({prev_two: [current]})  # {LongTensor: list containing 1 int}
                    elif t > 3:
                        # 前两个词已有记录，则将新的词添加到对应的键值列表中
                        if prev_two in trigrams[i]:  # add to list
                            trigrams[i][prev_two].append(current)
                        # 否则创建一个新的键值对
                        else:  # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                # 阻塞下一步使用已使用的三元组【作用是尽可能减小词的重复】
                prev_two_batch = seq[:, t - 2:t]
                # 掩码用以标记已使用的三元组
                mask = torch.zeros(logprobs.size(), requires_grad=False).to(logprobs.device)  # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i, j] += 1 # 已使用过的位置上索引置为1
                # Apply mask to log probs
                # logprobs = logprobs - (mask * 1e9)
                alpha = 2.0  # = 4
                logprobs = logprobs + (mask * -0.693 * alpha)  # ln(1/2) * alpha (alpha -> infty works best)

            # sample the next word
            if t == self.seq_length:  # skip if we achieve maximum length   # 如果已经达到最大长度，则跳过
                break
            # 根据不同采样策略（greedy/gumble/温度），进行下一个词的采样，得到某个词的索引it和对应概率【比如选择了词A】
            it, sampleLogprobs = self.sample_next_word(logprobs, sample_method, temperature)

            # stop when all finished
            if t == 0:
                unfinished = it != self.eos_idx
            else:
                # 已完成位置索引值填充为0
                it[~unfinished] = self.pad_idx  # This allows eos_idx not being overwritten to 0
                # 更新概率，将已完成位置的概率值置为0
                logprobs = logprobs * unfinished.unsqueeze(1).to(logprobs)
                unfinished = unfinished & (it != self.eos_idx)  # 通过是否为eos_idx，更新未完成位置的标记，判断序列是否已完成
            seq[:, t] = it  # 将当前生成的词，添加到对应位置上
            seqLogprobs[:, t] = logprobs    # 添加对应的概率值
            # quit loop if all sequences have finished  # 如果所有序列都已完成，则退出循环
            if unfinished.sum() == 0:
                break

        return seq, seqLogprobs


    def _forward(self, obj_f, grid_f, tag_f, seq, att_masks=None):
        # Attention: grid_f is a dict, included 'whole', 'four', 'nine', 'sixteen', 'twentyfive'
        # it needs to transform to a list to copy when generarating multiple captions

        if seq.ndim == 3:
            seq = seq.reshape(-1, seq.shape[2]) # (B * seq_per_img) * seq_len
            # seq[0] = (B * seq_per_img)

        # 先把obj、grid、tag传入projector，生成z即att_feats（可训练）
        att_feats = self.model.projector(obj_f, grid_f, tag_f)  # (b_s, ?, 512)
        # 进行特征预处理（有必要时会复制多个特征和掩码，以生成多个句子）
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks, seq)

        seq = seq.clone()
        # 将序列中的填充位置替换为 -1，以便在后续的计算中忽略这些填充位置
        seq[~seq_mask.any(-2)] = -1  # Make padding to be -1 (my dataloader uses 0 as padding)

        outputs = self.model(att_feats, seq)
        return outputs


    # 传输到decoder，获取下一步的概率，和state
    def get_logprobs_state(self, it, fc_feats, att_feats, p_att_feats, att_masks, state, output_logsoftmax=1):
        # 'it' contains a word index
        xt = self.embed(it) # xt = it

        output, state = self.core(xt, fc_feats, att_feats, p_att_feats, state, att_masks)
        if output_logsoftmax:
            logprobs = F.log_softmax(self.logit(output), dim=1)
        else:
            logprobs = self.logit(output)

        return logprobs, state

    # 根据当前输入和状态，生成序列的下一个单词，并更新状态
    # 【用于sample或多样性束搜索中】
    # 输入当前步的it，0，特征的第一个，memory，state[]即所选择的词，att_masks
    # 只用到了memory、state、att_masks，传输到decoder
    # 输出下一个词的概率，以及更新后的state
    def core(self, it, fc_feats, att_feats, memory, state, att_masks):
        """
        state = [ys.unsqueeze(0)]
        此处的memory指的是encoder的输出（论文中将其称为memory meshed）
        而state指的是decoder每个时间步解码器的状态
        此处的mask是指的注意力的mask？（用以屏蔽填充位置）？
        """
        if len(state) == 0:
            ys = it.unsqueeze(1)    # ->(b_s * 5, 1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        out = self.model.decoder(ys, memory, att_masks)
        return out[:, -1], [ys.unsqueeze(0)]    # state: [(1, b_s * 5, 1)]

    # # 此处是用旧的beamsearch方法，【覆盖】了AttModel中的_sample_beam方法，可暂时不管（普通的束搜索）
    # def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
    #     beam_size = opt.get('beam_size', 10)
    #     group_size = opt.get('group_size', 1)
    #     sample_n = opt.get('sample_n', 10)
    #     assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
    #
    #
    #     # 生成多个句子时复制多个att_feats
    #     att_feats, _, __, ___ = self._prepare_feature_forward(att_feats, att_masks)
    #
    #     seq, logprobs, seqLogprobs = self.model.beam_search(att_feats, self.seq_length, 0,
    #                                                         beam_size, return_probs=True, out_size=beam_size)
    #     seq = seq.reshape(-1, *seq.shape[2:])
    #     seqLogprobs = seqLogprobs.reshape(-1, *seqLogprobs.shape[2:])
    #
    #     # if not (seqLogprobs.gather(-1, seq.unsqueeze(-1)).squeeze(-1) == logprobs.reshape(-1, logprobs.shape[-1])).all():
    #     #     import pudb;pu.db
    #     # seqLogprobs = logprobs.reshape(-1, logprobs.shape[-1]).unsqueeze(-1).expand(-1,-1,seqLogprobs.shape[-1])
    #     return seq, seqLogprobs

    # 传入的fc是0（但有相应形状
    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        group_size = opt.get('group_size', 1)
        sample_n = opt.get('sample_n', 10)
        # assert sample_n == 1 or sample_n == beam_size // group_size, 'when beam search, sample_n == 1 or beam search'
        batch_size = fc_feats.size(0)

        # 此处p_fc=0, p_att是特征的第一个, pp_att是经过encoder编码的memory, p_att是井过encoder编码的注意力特征
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        # 用于存储序列和概率
        seq = fc_feats.new_full((batch_size * sample_n, self.seq_length), self.pad_idx, dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size * sample_n, self.seq_length, self.vocab_size + 1)
        # lets process every image independently for now, for simplicity
        # 用于存储生成的序列
        self.done_beams = [[] for _ in range(batch_size)]

        state = self.init_hidden(batch_size)

        # first step, feed bos
        # 全0张量，用于存储<bos>标记  # (batch_size)
        it = fc_feats.new_full([batch_size], self.bos_idx, dtype=torch.long)
        # 获取下一步的概率和状态
        logprobs, state = self.get_logprobs_state(it, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, state)

        # 复制beam_size份
        p_fc_feats, p_att_feats, pp_att_feats, p_att_masks = utils.repeat_tensors(beam_size,
                                                                                  [p_fc_feats, p_att_feats,
                                                                                   pp_att_feats, p_att_masks]
                                                                                  )
        # 多样性束搜索（此处默认返回10个不同组的结果）
        self.done_beams = self.beam_search(state, logprobs, p_fc_feats, p_att_feats, pp_att_feats, p_att_masks, opt=opt)
        for k in range(batch_size):
            if sample_n == beam_size:  # 每个束都是一个样本
                for _n in range(sample_n):  # 遍历每个束，存储seq和概率
                    seq_len = self.done_beams[k][_n]['seq'].shape[0]
                    seq[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['seq']
                    seqLogprobs[k * sample_n + _n, :seq_len] = self.done_beams[k][_n]['logps']
            else:  # 否则取第一个beam作为样本
                seq_len = self.done_beams[k][0]['seq'].shape[0]
                seq[k, :seq_len] = self.done_beams[k][0]['seq']  # the first beam has highest cumulative score
                seqLogprobs[k, :seq_len] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods

        return seq, seqLogprobs