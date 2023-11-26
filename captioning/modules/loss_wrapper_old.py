import pdb

import torch
from .rewards import get_scores, get_self_cider_scores, get_SCST_reward
from .ghm_loss import GHMC
import torch.nn as nn
import numpy as np

from torch.autograd import Variable

# 计算语言模型XE损失：GHMC损失or负对数似然损失
# 等同于负对数似然损失NLLLoss操作
class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()
        if isinstance(opt, dict):
            self.use_ghm = opt.get('use_ghmloss', False)
        else:
            self.use_ghm = getattr(opt, 'use_ghmloss', False)
        # 梯度平衡损失函数，通过引入梯度倒数的指数平均gradient harmonized，平衡各个样本的梯度贡献
        if self.use_ghm:
            self.ghmloss = GHMC()

    def forward(self, input, target, mask):
        """

        :param input:   (batch_size, seq_len, vocab_size)
        :param target:  (batch_size, seq_len)
        :param mask:    (batch_size, seq_len)
        :return:
        """
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        # 获取target对应位置上的概率值
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2)
        if self.use_ghm:
            output = self.ghmloss(output, target, mask)
        else:
            output = output * mask
        # Average over each token
        # 计算求和总损失，平均到每个token上
        # 概率越大，负对数似然损失越小
        output = torch.sum(output) / torch.sum(mask)

        # 计算序列语言模型的负对数似然损失
        return output

# 计算强化学习损失：- reward * logP * mask
class RewardCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward):
        """
        input: 模型输出概率，(b*5, seq_l, v_s)
        seq: 生成序列，(b*5, seq_l)
        reward: 奖励，(b*5, seq_l)
        """
        # input通过gather进行索引，得到模型的输出值
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)    # ->(b*5, seq_l)    # 选定词对应的概率
        # 按索引对齐
        input = input.reshape(-1)   # 展平(b*5*seq_l)
        reward = reward.reshape(-1)
        # 考虑序列的填充符，只需考虑前L-1个位置，最后一个位置对应的损失值为0
        mask = (seq>0).to(input)
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        # 根据序列长度进行平均，得到最终损失值
        # 对应位置元素相乘后，求和，最终得到一个标量，作为模型最终损失值
        output = - input * reward * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

# 计算模型输出和平滑后的标签之间的差异
# 即在标签的每一维上，添加一个随机噪音，更加泛化
# 将hard_label转化为soft_habel
class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)  # 计算KLDiv损失
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing  # 平滑因子
        # self.size = size
        self.true_dist = None

    def forward(self, input, target, mask):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        # truncate to the same size
        # 截断，使长度与input第1维相同
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]

        # 展平
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()  # 复制
        # true_dist.fill_(self.smoothing / (self.size - 2)
        # 将所有元素赋值为平滑因子/(size-1)
        true_dist.fill_(self.smoothing / (self.size - 1))
        # 将true_dist中对应target位置的元素，替换为置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        # 用平滑后的标签分布，计算得到KL散度损失，再用掩码加权平均
        return (self.criterion(input, true_dist).sum(1) * mask).sum() / mask.sum()

class LossWrapper(nn.Module):
    def __init__(self, model, opt):
        super(LossWrapper, self).__init__()

        self.opt = opt
        self.model = model
        self.train_sample_n = getattr(opt, 'train_sample_n', 5)

        self.xe_crit = LanguageModelCriterion(opt)
        self.rl_crit = RewardCriterion(opt)
        if opt.label_smoothing > 0:
            self.crit = LabelSmoothing(smoothing=opt.label_smoothing)
        else:
            self.crit = LanguageModelCriterion(opt)

        # self._loss = {}

    def forward(self, obj_f, grid_f, tag_f, labels, masks, tags, gts, gt_indices,
                SCST_flag):
        # labels：(b_s, seq_per_img, seq_l)，用于decoder生成
        # mask是labels的mask
        # gts是数据集中所有的句子，用于计算强化学习奖励
        opt = self.opt
        self.seq_length = getattr(opt, 'max_length', 20) or opt.seq_length
        out = {}

        # self.XE_flag = XE_flag
        self.SCST_flag = SCST_flag
        self.base_type = getattr(opt, 'base_type', 'average')  # gt or greedy

        # 普通XE训练
        if not self.SCST_flag:
            # model_outputs
            sample_logprobs = self.model(obj_f, grid_f, tag_f, labels[..., :-1])    # 给出前缀，预测最后一个词
            # 此模式下model_outputs只有一个，即为sample_logprobs
            loss = self.xe_crit(sample_logprobs, labels[..., 1:], masks[..., 1:])   # 标签是从第二个词开始的

        # SCST强化学习训练
        elif self.SCST_flag:
            # if self.base_type == 'gt':  # No baseline   # 纯计算分数
            #
            #     gen_result, sample_logprobs = self.model(obj, grid, tag, gts, masks)
            #     baseline = gts
            #
            #     scores = get_scores(baseline, gen_result, opt)  # 计算CIDEr、Bleu得分
            #     if getattr(self.opt, 'self_cider_reward_weight', 0) > 0:    # 计算self-CIDEr得分
            #         _scores = get_self_cider_scores(gts, gen_result, opt)
            #         _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1)
            #         # _scores = _scores.expand_as(scores - 1)
            #         _scores = _scores.expand_as(scores)
            #         scores += self.opt.self_cider_reward_weight * _scores   # 得分直接作奖励
            #     loss = self.rl_crit(sample_logprobs, gen_result, scores)    # 求得loss

            if self.base_type == 'greedy':    #  可束搜索，可随机采样
                self.model.eval()
                with torch.no_grad():
                    # 贪婪搜索获取greedy的句子   # 【不需要传入gts？】
                    greedy_res, sample_logprobs_greedy = self.model(obj_f, grid_f, tag_f,
                        opt={'sample_method': 'greedy',
                             'beam_size': 1},
                        mode='sample')
                self.model.train()
                # 进行采样，生成5个句子           # 【不需要传入gts？】
                gen_result, sample_logprobs = self.model(obj_f, grid_f, tag_f,
                        opt={'sample_method': 'sample',
                             'beam_size': 1,
                             'sample_n': self.num_sample_captions}, # 5
                        mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()] # 所有的句子，用gt_indices索引来查找
                # for i,v in enumerate(gts):
                #     delta = self.seq_length - len(v)
                #     if delta > 0:
                #         gts[i] = np.concatenate([v, np.zeros(delta, dtype=np.int64)])
                #     else:
                #         gts[i] = v[:self.seq_length]

                # 获取自我批判奖励
                reward = get_SCST_reward(gts, gen_result, greedy_res, tags, self.opt)  # (b*5, seq_l)
                reward = torch.from_numpy(reward).to(sample_logprobs)

                # greedy只有一个句子，无法作为baseline计算Self-CIDEr分数
                # 以gts为基准计算Self-CIDEr奖励

                if getattr(self.opt, 'self_cider_reward_weight', 1.0) > 0:
                    _scores1 = get_self_cider_scores(gts, gen_result, self.opt)    # 计算生成句子的Self-CIDEr
                    _scores2 = get_self_cider_scores(gts, gts, self.opt)           # 计算gt的Self-CIDEr，用作baseline
                    _scores = _scores1 - _scores2   # shape: b
                    _scores = torch.from_numpy(_scores).type_as(reward).expand(_scores.size(0), self.num_sample_captions).view(-1, 1) # (b*5, 1)
                    # _scores = _scores.expand_as(scores - 1)
                    _scores = _scores.expand_as(reward) # (b*5, seq_l)
                    reward += self.opt.self_cider_reward_weight * _scores   # 奖励：CIDEr+BLEU+Self-CIDEr
                loss = self.rl_crit(sample_logprobs, gen_result, reward)    # 将reward与logprobs相乘，得到loss
                out['reward'] = reward[:,0].mean()


            elif self.base_type == 'average':    # baseline只进行随机采样，与剩余的其他的均值比
                # Noted that only used when random sample in baseline
                # it will fail when used beam search

                gen_result, sample_logprobs = self.model(obj_f, grid_f, tag_f,
                                                         opt={'sample_method': 'sample',
                                                              'beam_size': 1,
                                                              'sample_n': self.train_sample_n},    # 5
                                                         mode='sample')
                gts = [gts[_] for _ in gt_indices.tolist()] # 所有的句子，用gt_indices索引来查找    # 都是tokens
                # print(type(gts), np.array(gts).shape)
                # for i,v in enumerate(gts):
                #     delta = self.seq_length - len(v)
                #     if delta > 0:
                #         gts[i] = np.concatenate([v, np.zeros(delta, dtype=np.int64)])
                #     else:
                #         gts[i] = v[:self.seq_length]
                scores = get_scores(gts, gen_result, tags, opt)   # shape: (b*5, 1)
                # tags: (b_s, topk)
                scores = torch.from_numpy(scores).type_as(sample_logprobs).view(-1, self.train_sample_n)  # (b*5, 1) -> (b, 5)

                if self.opt.base_range == 'avg':
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

                scores = scores - baseline  # shape: （b, 5)
                print('reward_1', scores.mean())
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

                    # # gen+gt的多样性得分
                    # _gts = torch.from_numpy(np.array(gts).astype(np.int32)).type_as(gen_result)
                    # pads = torch.zeros((_gts.shape[0], _gts.shape[1], 4)).to(_gts) # 4是长度差值
                    # _gts = torch.cat([_gts, pads], dim=-1) # (b, 5, 20)
                    # _gen = gen_result.unsqueeze(0).expand_as(_gts) # (b, 5, 20)
                    # mix = torch.concat([_gts, _gen], dim=-2) # 【gts+gen_result的结果】
                    # mix = mix.squeeze(0)
                    # _scores2 = get_self_cider_scores(gts, mix, self.opt)
                    # print(_scores2)

                    # r(y-b)
                    _scores = _scores1 - _scores_gts   # shape: b
                    _scores = torch.from_numpy(_scores).type_as(scores).view(-1, 1) # (b, 1)
                    print('Self-CIDEr scores:', _scores.mean())

                    _scores = _scores.expand_as(scores) # (b, 5)
                    scores += self.opt.self_cider_reward_weight * _scores   # shape: （b, 5)

                reward = scores.view(-1, 1).expand_as(gen_result) # (b*5, 1) -> (b*5, seq_l)
                loss = self.rl_crit(sample_logprobs, gen_result, reward)    # 将scores与logprobs相乘，得到loss
                out['reward'] = reward[:,0].mean()

        out['loss'] = loss
        return out



        # if self.num_sample_captions > 1:
        #     gen_results_list = []
        #     loss_temp = Variable(torch.FloatTensor([0])).cuda()
        #     # 待思考奖励函数用矩阵算还是直接单个句子算，下面先按单个句子算
        #     loss_ratio = Variable(torch.zero(self.batch_size)).cuda
        #     self._loss['avg_reward'] = 0
        #     self._loss['cider_greedy'] = 0
        #
        #     for i_num in range(self.num_sample_captions):
        #         if 'greedy_res' not in locals():
        #             greedy_res, _ = self.model.sample(
        #                 *utils.var_wrapper([obj, grid, tag], volatile=True),
        #                 opt={'sample_max': 1})
        #         gen_result, sample_logprobs = self.model.sample(
        #             obj, grid, tag, opt={'sample_max': 0})
        #         gen_masks = torch.cat([Variable(gen_result.data.new(gen_result.size(0), 2). fill_(1).float()),
        #                                (gen_result > 0).float()[:,:-1]], 1) # 第1维上拼接，去除最后一个结束符
        #         gen_results_list.append(gen_result)
        #
        #         # 使用自我批判训练
        #         # 【待改，加入NSC、两种RMR以及混合，共6种】
        #         # if self.self_critical >= 1:
        #         #     reward, cider_greedy = rewards.get_self_critical_reward(data, gen_result, greedy_res)
        #         # else:
        #         #     reward, cider_greedy = rewards.get_cider_reward(data, gen_result, greedy_res)
        #
        #         # 默认使用SCST
        #         reward, cider_greedy = get_SCST_reward(data, gen_result, greedy_res, opt)
        #
        #         # 添加平均奖励和贪婪搜索的CIder得分
        #         self._loss['avg_reward'] += reward.mean()
        #         self._loss['cider_greedy'] += cider_greedy
        #
        #         # 计算生成损失
        #         loss_cap = sample_logprobs * utils.var_wrapper(-reward.astype('float32')).unsqueeze(1)*(gen_masks[:,1:].detach())
        #         loss_cap = loss_cap.sum() / gen_masks[:,1:].data.float().sum()
        #         # 累计多个生成句子的损失
        #         loss_temp += loss_cap
        #         # print(loss_ratio.shape, sample_logprobs.shape, gen_masks[:, 1:].shape)
        #         # 每个句子中，每个非填充词的概率，之和，的平均值
        #         # 用作多样性惩罚
        #         loss_ratio += torch.mul(sample_logprobs, gen_masks[:, 1:].detach()).sum(1) / gen_masks[:, 1:].sum(1)
        #
        #     # 平均每个句子的奖励和贪婪分数
        #     self._loss['avg_reward'] /= self.num_sample_captions
        #     self._loss['cider_greedy'] /= self.num_sample_captions
        #
        #     # # 根据多样性，输入生成的n个句子，计算ratio
        #     # if self.diversity_metric == 'LSA':
        #     #     ratio = rewards.get_lsa_reward(gen_results_list)
        #     # elif self.diversity_metric == 'selfcider':
        #     #     ratio = rewards.get_self_cider_reward_parallel(gen_results_list)
        #     # else:
        #     #     raise IOError
        #
        #     # 多个句子的Self-CIDEr得分
        #     ratio, ratio_gradient = rewards.get_self_cider_reward_parallel(gen_results_list)
        #
        #     # 计算loss
        #     loss_ratio = loss_ratio * utils.var_wrapper(ratio_gradient.astype('float32'))
        #     # 加权总loss
        #     loss += self.CIDEr_weight * loss_temp / self.num_sample_captions + \
        #         self.Div_weight * loss_ratio.sum() / (loss_ratio.shape[0] * self.num_sample_captions)
        #     self._loss['ratio'] = ratio.mean()
        #
        #
        # self._loss['loss'] = loss.data[0]
        # return loss

    # def sample(self, fc_feats, att_feats, att_masks, opt={}):
    #     return self.model.sample(fc_feats, att_feats, att_masks, opt)
    #
    # def loss(self):
    #     out = {}
    #     out.update(self._loss)
    #
    #     return out