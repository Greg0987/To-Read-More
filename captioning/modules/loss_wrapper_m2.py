import pdb
import sys

import torch
from . import losses
from .rewards import get_scores, get_self_cider_scores, get_SCST_reward
from .ghm_loss import GHMC
import torch.nn as nn
import numpy as np

from torch.autograd import Variable


# 将loss包在模型输出里
class LossWrapperM2(nn.Module):
    def __init__(self, model, opt):
        super(LossWrapperM2, self).__init__()

        self.opt = opt
        self.model = model
        self.train_sample_n = getattr(opt, 'train_sample_n', 5)
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length

        self.xe_crit = losses.LanguageModelCriterion(opt)
        self.rl_crit = losses.RewardCriterion(opt)
        if opt.label_smoothing > 0:
            self.crit = losses.LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = losses.LanguageModelCriterion(opt)

        # self._loss = {}

    def forward(self, obj_f, labels, masks, tags, gts, gt_indices,
                SCST_flag):
        # labels：(b_s, seq_per_img, seq_l)，用于decoder生成
        # mask是labels的mask
        # gts是数据集中所有的句子，用于计算强化学习奖励
        opt = self.opt

        out = {}

        # self.XE_flag = XE_flag
        self.SCST_flag = SCST_flag
        self.base_type = getattr(opt, 'base_type', 'average')  # gt or greedy

        # 普通XE训练
        if not self.SCST_flag:
            # model_outputs
            sample_logprobs = self.model(obj_f, labels[..., :-1])    # 给出前缀，预测最后一个词
            # 此模式下model_outputs只有一个，即为sample_logprobs
            loss = self.xe_crit(sample_logprobs, labels[..., 1:], masks[..., 1:])   # 标签是从第二个词开始的

        # SCST强化学习训练
        elif self.SCST_flag:
            # 贪婪搜索，只生成一个base句子，无法加入self-cider奖励
            if self.base_type == 'greedy':    #  可束搜索，可随机采样
                self.model.eval()
                with torch.no_grad():
                    # 贪婪搜索获取greedy的句子   # 【不需要传入gts？】
                    greedy_res, sample_logprobs_greedy = self.model(obj_f,
                        opt={'sample_method': 'greedy',
                             'beam_size': 1},
                        mode='sample')
                self.model.train()
                # 进行采样，生成5个句子           # 【不需要传入gts？】
                gen_result, sample_logprobs = self.model(obj_f,
                        opt={'sample_method': 'sample',
                             'beam_size': 1,
                             'sample_n': self.train_sample_n}, # 5
                        mode='sample')
                # 真实值
                gts = [gts[_] for _ in gt_indices.tolist()] # 所有的句子，用gt_indices索引来查找
                # 获取自我批判奖励
                reward = get_SCST_reward(gts, gen_result, greedy_res, tags, self.opt)  # (b*5, seq_l)
                reward = torch.from_numpy(reward).to(sample_logprobs)
                loss = self.rl_crit(sample_logprobs, gen_result, reward)    # 将reward与logprobs相乘，得到loss
                out['reward'] = reward[:,0].mean()

            # 束搜索，M2中采用的方法
            elif self.base_type == 'beamsearch':
                # 用beam search获取5个句子
                # 生成序列(5n, seq_l)和logprobs(5n, seq_l, vocab_size)
                gen_result, sample_logprobs = self.model(obj_f,
                                                            opt={'sample_method': 'beam_search',
                                                                    'beam_size': 5,
                                                                    'sample_n': self.train_sample_n},    # 5
                                                            mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()]

                # 返回5n个分数
                scores = get_scores(gts, gen_result, tags, opt)   # shape: (b*5, 1)
                scores = torch.from_numpy(scores).type_as(sample_logprobs).view(-1, self.train_sample_n)  # ->(b, 5)

                reward_baseline = torch.mean(scores, dim=-1, keepdim=True)  # (b, 1)
                # 根据生成序列，取出对应的可能性->(b, 5)

                log_prob = sample_logprobs.gather(2, gen_result.unsqueeze(2)).squeeze(2)    # ->(b * 5, seq_l)
                log_prob = log_prob.view(-1, self.train_sample_n, self.seq_length)  # ->(b, 5, seq_l)

                reward = scores - reward_baseline   # (b, 5)

                loss = -torch.mean(log_prob, -1) * (scores - reward_baseline)  # (b, 5)

                out['reward'] = scores.mean()


            # 其余采样候选的均分作为baseline【NSC】
            # 所有样本的范围中值作为baseline【MSC】
            elif self.base_type == 'average':    # baseline进行随机采样
                # 用所有的候选样本的范围中值  #即用get_scores
                # Noted that only used when random sample in baseline
                # it will fail when used beam search

                gen_result, sample_logprobs = self.model(obj_f,
                                                         opt={'sample_method': 'sample',
                                                              'beam_size': 1,
                                                              'sample_n': self.train_sample_n},    # 5
                                                         mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()] # 所有的句子，用gt_indices索引来查找    # 都是tokens

                # 返回5n个分数
                scores = get_scores(gts, gen_result, tags, opt)   # shape: (b*5, 1)
                # tags: (b_s, topk)
                scores = torch.from_numpy(scores).type_as(sample_logprobs).view(-1, self.train_sample_n)  # (b*5, 1) -> (b, 5)

                if self.opt.base_range == 'avg':    # 均值
                    baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)   # shape: （b, 5)
                elif self.opt.base_range == 'mid':  # 中值1
                    baseline = (scores.max(1, keepdims=True)[0] + scores.min(1, keepdims=True)[0]) / 2  # shape: （b, 1)
                elif self.opt.base_range == 'mid2':  # 中值2
                    scores_ = scores.new_full((*scores.shape, scores.shape[1] - 1), 0.)
                    for i in range(scores.shape[1]):
                        scores_[:, i, :i] = scores[:, :i]
                        scores_[:, i, i:] = scores[:, i + 1:]
                    baseline = (scores_.max(-1)[0] + scores_.min(-1)[0]) / 2    # shape: （b, 5)
                elif self.opt.base_range == 'mix':  # 混合
                    avg = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1)
                    mid = (scores.max(1, keepdims=True)[0] + scores.min(1, keepdims=True)[0]) / 2
                    scores_avg = scores - avg
                    scores_mid = scores - mid
                    baseline = torch.where(scores_mid.abs() > scores_avg.abs(), mid, avg)   # shape: （b, 5)
                else:
                    raise Exception(self.opt.base_range, 'are not supported')

                scores_ = scores - baseline  # shape: （b, 5)
                print('reward_1: %.4f' % scores_.mean().item())

                # print('=====scores=====', scores, scores.shape)
                if getattr(self.opt, 'self_cider_reward_weight', 1.0) > 0:
                    # 生成句子的多样性得分
                    _scores1 = get_self_cider_scores(gts, gen_result, self.opt) # shape: b
                    # print("=====scores_1=====", _scores1, _scores1.shape)

                    # gt的多样性得分
                    # print(np.array(gts).shape)
                    _gts = torch.from_numpy(np.array(gts).astype(np.int32))
                    # _gts = _gts.squeeze(0)
                    _gts = _gts.view(-1, _gts.shape[-1])
                    # print(gen_result.shape, _gts.shape)
                    _scores_gts = get_self_cider_scores(gts, _gts, self.opt) # shape: b
                    # print("=====scores_gts=====", _scores_gts, _scores_gts.shape)

                    # r(y-b)
                    _scores = _scores1 - _scores_gts   # shape: b
                    _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1) # (b, 1)
                    print('SCreward: %.4f'% _scores.mean().item())

                    _scores = _scores.expand_as(scores) # (b, 5)
                    scores_ += self.opt.self_cider_reward_weight * _scores   # shape: （b, 5)

                reward = scores_.view(-1, 1).expand_as(gen_result) # (b*5, 1) -> (b*5, seq_l)
                loss = self.rl_crit(sample_logprobs, gen_result, reward)    # 将scores与logprobs相乘，得到loss
                out['reward'] = scores.mean()

            else:
                raise Exception("Reinforcement base type not supported: {}".format(opt.base_type))

        try:
            out['loss'] = loss
        except RuntimeError as e:
            if "out of memory" in str(e):
                sys.exit('Out of Memory')
            else:
                raise e
        return out