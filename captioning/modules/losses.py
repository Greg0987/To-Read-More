import torch
import torch.nn as nn

# 将输出结果，和对应reward进行计算（即强化学习公式），返回loss
class RewardCriterion(nn.Module):
    def __init__(self, opt):
        super(RewardCriterion, self).__init__()

    def forward(self, input, seq, reward, reduction='mean'):
        """

        :param input:   模型输出的logits，(b_s, seq_l, vocab_size)
        :param seq:     生成序列，(b_s, seq_l)
        :param reward:  奖励，(b_s, seq_l)
        :param reduction:
        :return:
        """
        N,L = input.shape[:2]
        # input通过gather函数按seq的索引取出对应的值
        input = input.gather(2, seq.unsqueeze(2)).squeeze(2)
        # 按索引对齐
        input = input.reshape(-1)
        reward = reward.reshape(-1)
        mask = (seq>0).to(input)
        # 考虑序列的填充符，只需考虑前L-1个位置，最后一个位置对应的损失值为0
        mask = torch.cat([mask.new(mask.size(0), 1).fill_(1), mask[:, :-1]], 1).reshape(-1)
        # 根据序列长度进行平均，得到最终损失值
        output = - input * reward * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output

# 语言模型损失
class LanguageModelCriterion(nn.Module):
    def __init__(self, opt):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask, reduction='mean'):
        if target.ndim == 3:
            target = target.reshape(-1, target.shape[2])
            mask = mask.reshape(-1, mask.shape[2])
        N,L = input.shape[:2]
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)].to(input)

        # 获取target对应位置上的logits
        output = -input.gather(2, target.unsqueeze(2)).squeeze(2) * mask

        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        # Average over each token
        # 计算求和总损失，平均到每个token上
        # 概率越大，负对数似然损失越小
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output



# 计算模型输出和平滑后的标签之间的差异
# 即在标签的每一维上，添加一个随机噪音，更加泛化
# 将hard_label转化为soft_habel
class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size=0, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False, reduce=False)
        # self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing  # 平滑因子
        # self.size = size
        self.true_dist = None
        
    def forward(self, input, target, mask, reduction='mean'):
        N,L = input.shape[:2]
        # truncate to the same size
        # 截断，使长度与input第1维相同
        target = target[:, :input.size(1)]
        mask =  mask[:, :input.size(1)]

        # 战平
        input = input.reshape(-1, input.size(-1))
        target = target.reshape(-1)
        mask = mask.reshape(-1).to(input)

        # assert x.size(1) == self.size
        self.size = input.size(1)
        # true_dist = x.data.clone()
        true_dist = input.data.clone()
        # true_dist.fill_(self.smoothing / (self.size - 2))
        # 将所有元素赋值为平滑因子/(size-1)
        true_dist.fill_(self.smoothing / (self.size - 1))
        # 将true_dist中对应target位置的元素，替换为置信度
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        # true_dist[:, self.padding_idx] = 0
        # mask = torch.nonzero(target.data == self.padding_idx)
        # self.true_dist = true_dist
        output = self.criterion(input, true_dist).sum(1) * mask
        
        if reduction == 'none':
            output = output.view(N,L).sum(1) / mask.view(N,L).sum(1)
        elif reduction == 'mean':
            output = torch.sum(output) / torch.sum(mask)

        return output