
# ========== version 2 ==========
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
from collections import OrderedDict
import torch

import string

punctuation_list = string.punctuation
escapes = ''.join([chr(char) for char in range(0, 32)])
import pprint
import sys
import os
import h5py

# try:
# sys.path.append(os.path.dirname(__file__))
sys.path.append(os.getcwd())
from cider.pyciderevalcap.ciderD.ciderD import CiderD
from cider.pyciderevalcap.cider.cider import Cider
from coco_caption.pycocoevalcap.bleu.bleu import Bleu
# except:
#     print('cider or coco-caption missing')
from ..utils import misc as utils

CiderD_scorer = None
Cider_scorer = None
Bleu_scorer = None
#CiderD_scorer = CiderD(df='corpus')

def init_scorer(cached_tokens):
    global CiderD_scorer
    CiderD_scorer = CiderD_scorer or CiderD(df=cached_tokens)
    global Cider_scorer
    Cider_scorer = Cider_scorer or Cider(df=cached_tokens)
    global Bleu_scorer
    Bleu_scorer = Bleu_scorer or Bleu(4)

def array_to_str(arr):
    out = ''
    for i in range(len(arr)):
        out += str(arr[i]) + ' '
        if (arr[i] == 0).any():
            break
    return out.strip()

# 返回强化学习奖励
def get_SCST_reward(data_gts, gen_result, baseline, tags, opt):
    """
    :param data_gts: 真实标注                           # (b, 5, seq_l)
    :param gen_result: 生成的句子：跟gt进行比较得出分数    # (b*5, seq_l)
    :param baseline: 用于比较的基线：跟gt进行比较得出分数    # (b, seq_l)    # 此处为greedy_res
    :param opt: 其他参数
    :return:
    """
    batch_size = len(data_gts)  # b_s，即图片数n
    gen_result_size = gen_result.shape[0]   # b*5，即总条数5n条
    seq_per_img = gen_result_size // len(data_gts) # gen_result_size = batch_size * seq_per_img
    assert baseline.shape[0] == batch_size  # 确保贪婪搜索条数==图片数
    # (b, seq_l)

    res = OrderedDict()
    gen_result = gen_result.data.cpu().numpy()
    baseline = baseline.data.cpu().numpy()
    for i in range(gen_result_size):    # 前面n*5条为生成句子
        res[i] = [array_to_str(gen_result[i])]
    for i in range(batch_size):         # 后面的n条为baseline句子(贪婪搜索)
        res[gen_result_size + i] = [array_to_str(baseline[i])]

    gts = OrderedDict()
    for i in range(len(data_gts)):
        # 每个gts包含多个句子，即i张图片含j个句子
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(len(res))]    # 用于CIDEr
    res__ = {i: res[i] for i in range(len(res_))}                           # 用于Bleu
    # 生成的5条句子对应同一个gt组，如句子1-gt1组，句子2-gt1组，句子3-gt1...
    gts_ = {i: gts[i // seq_per_img] for i in range(gen_result_size)}
    # 把贪婪搜索的句子也同gt对应起来
    gts_.update({i + gen_result_size: gts[i] for i in range(batch_size)})

    # ==========计算CIDEr奖励，用res_格式==========
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts_, res_)   # 5n + n个分数
        print('Cider scores:', _)
    else:
        cider_scores = 0

    # ==========计算Bleu奖励，用res__格式==========
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts_, res__) # 5n + n个分数
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores:', _[3])
    else:
        bleu_scores = 0

    # scores.shape: (b_s * 5 + b_s, 1)
    # res.shape: (n * 5 + n, seq_l)

    # ==========计算Cov奖励==========
    # 标签覆盖率是指，句子tokens中所含的标签tokens数，占句子tokens的比例
    # 计算标签覆盖率的函数
    tgs = OrderedDict()
    for i in range(batch_size):
        tgs[i] = [array_to_str(tags[i])]    # n
    # 同样把标签，与gen+baseline对齐 # 此处标签也是token id
    tgs_ = {i: tgs[i//seq_per_img] for i in range(gen_result_size)}         # 前5n条句子，对应的标签
    tgs_.update({gen_result_size + i: tgs[i] for i in range(batch_size)})   # 后n条贪婪搜索的句子，对应的标签

    # Cov分数
    if opt.cov_reward_weight > 0:
        cov_scores = np.zeros_like(cider_scores)    # (5n + n, 1)
        # cov_scores = np.zeros((gen_result_size + batch_size, 1))
        for i in range(gen_result_size + batch_size):
            _sentence = [_.item() for _ in res__[i] if _ != 0]   # _sentence需要去除为0的token，因为这些token是pad的
            # _sentence_set = set(_.item() for _ in res__[i] if _ != '0')   # _sentence需要去除为'0'的token，因为这些token是pad的
            tags_set = set(_.item() for _ in tgs_[i])
            common_words_count = [_ for _ in _sentence if _ in tags_set]
            cov_scores[i] = len(common_words_count) / len(_sentence) \
                if len(_sentence) > 0 else 0
        print('Cov scores:', np.mean(cov_scores))
    else:
        cov_scores = 0


    # noted: there are some tokens which is pad in gen_result and baseline, which will affect the length of sentence
    # noted: CL奖励可为负数，因为生成的句子可能比baseline短
    # def CLength(gen_result, baseline):
    #     gen_result = gen_result[gen_result > 0] # 需要确保pad是0，包括eos、bos、pad
    #     baseline = baseline[baseline > 0]
    #     return len(gen_result) - len(baseline)

    # ==========计算CL奖励==========
    # The cl_scores is the difference between the lenth of generated sentence and the baseline sentence
    # generated sentence type: (b*5, seq_l)
    # baseline sentence type: (b, seq_l)
    if opt.cl_reward_weight > 0:
        cl_expected = opt.cl_expected
        cl_scores = np.zeros_like(cider_scores) # (6n ,1)
        # cl_scores = np.zeros((len(res), 1))
        for i in range(gen_result_size + batch_size):
            _sentence = [_ for _ in res__[i].split() if _ != '0']
            cl_res = len(_sentence)
            cl_scores[i] = (cl_res - cl_expected) / cl_expected
        print('CL scores:', np.mean(cl_scores))
    else:
        cl_scores = 0

    scores = opt.cider_reward_weight * cider_scores +\
             opt.bleu_reward_weight * bleu_scores +\
             opt.cov_reward_weight * cov_scores +\
             opt.cl_reward_weight * cl_scores

    # 计算reward (y^ - b)
    # scores前半：(n*5, 1) -> (n, 5)
    # scores后半：(n, 1) -> (n ,5)  # 进行了复制
    # 生成的5个句子，都是减去同一个baseline的分数
    scores = scores[:gen_result_size].reshape(batch_size, seq_per_img) -\
             scores[-batch_size:][:, np.newaxis]    # 数组扩展成相同形状

    rewards = np.repeat(scores[:, np.newaxis], gen_result.shape[1], 1)  # 重复seq_l次
    # 此时rewards变为(5n, seq_l)，与生成句子有相同的shape

    return rewards

# 返回Cider和Bleu分数【每个句子都会有1个得分，即一张图像对应5个得分】
# 用于baseline=='averange'
# 返回5n个分数
def get_scores(data_gts, gen_result, tags, opt):
    """

    :param data_gts:    # 真是标注n * 5条句子，(n, 5, seq_l)
    :param gen_result:  # 生成结果n * 5条句子，(n*5, seq_l)
    :param tags:        # n张图像的标签，(n, topk)
    :param opt:
    :return:
    """
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img   # n * 5条句子
    seq_per_img = batch_size // len(data_gts)

    tags_num = opt.tags_num

    vocab = opt.vocab

    res = OrderedDict()

    gen_result = gen_result.data.cpu().numpy()  # (n * 5, seq_l)
    for i in range(batch_size): # shape: 5n
        res[i] = [array_to_str(gen_result[i])]  # 5n条句子

    gts = OrderedDict()
    for i in range(len(data_gts)):  # shape: n
        gts[i] = [array_to_str(data_gts[i][j]) for j in range(len(data_gts[i]))]    # n*5条真实标注

    res_ = [{'image_id': i, 'caption': res[i]} for i in range(batch_size)]  # shape: 5n
    res__ = {i: res[i] for i in range(batch_size)}
    gts = {i: gts[i // seq_per_img] for i in range(batch_size)} # shape: 5n
    # 每条真实标注句子重复5次，以对应res

    print("-" * 20)

    # ==========计算CIDEr分数==========
    if opt.cider_reward_weight > 0:
        _, cider_scores = CiderD_scorer.compute_score(gts, res_)
        print('Cider scores: %.4f' % _)
        # _:(1), cider_scores:(5n)
    else:
        cider_scores = 0

    # ==========计算Bleu分数==========
    if opt.bleu_reward_weight > 0:
        _, bleu_scores = Bleu_scorer.compute_score(gts, res__)
        bleu_scores = np.array(bleu_scores[3])
        print('Bleu scores: %.4f' % _[3])
    else:
        bleu_scores = 0

    # ==========计算Cov分数==========
    # 【考虑分母为句子长度or标签数量】
    tgs = OrderedDict()
    for i in range(len(tags)):  # shape: n
        tgs[i] = tags[i].tolist()   # n张标签图像的200个标签
    tgs_ = {i: tgs[i // seq_per_img] for i in range(batch_size)}  # shape: 5n

    if opt.cov_reward_weight > 0:
        cov_scores = np.zeros_like(cider_scores)    # (5n)
        # cov_scores = np.zeros((batch_size, 1))
        for i in range(batch_size): # 5n
            # print(res__[i][0], type(res__[i][0]))
            # res__[i]是list,['1', '5380', '1270', '119', '1', '37', '1183', '28', '49', '179', '2574', '2571']
            # res__[i][0]是str, 1 5380 1270 119 1 37 1183 28 49 179 2574 2571
            _sentence = [int(_) for _ in res__[i][0].split() if _ != '0']
            # _sentence_set = set(word.item() for word in res__[i] if word != '0')
            tags_set = set(tag for tag in tgs_[i][:tags_num])
            common_words_count = [_ for _ in _sentence if _ in tags_set]
            # print(tags_set, common_words_count)
            cov_scores[i] = len(common_words_count) / len(_sentence) \
                if len(_sentence) > 0 else 0
        print('Cov scores: %.4f' % np.mean(cov_scores))
    else:
        cov_scores = 0

    # ==========计算CL分数==========
    # 【考虑是否需要与baseline句子作差值】
    if opt.cl_reward_weight > 0:
        cl_expected = opt.cl_expected
        cl_scores = np.zeros_like(cider_scores)
        # cl_scores = np.zeros((batch_size, 1))
        for i in range(batch_size): # 5n
            cl_res = max(0, len(res__[i][0].split()) - 1)  # 每个res句子的长度
            # cl_gts = sum([len(sentence.split()) for sentence in gts[i]]) / len(gts[i])  # 平均每个gts句子的长度
            # cl_scores[i] = (cl_res - cl_gts) / cl_gts
            cl_scores[i] = (cl_res - cl_expected) / cl_expected
            # cl_scores[i] = cl_res # 此处直接用生成句子的长度作为奖励，没有计算与gts的长度差值
        print('CL scores: %.4f' % np.mean(cl_scores))
    else:
        cl_scores = 0

    scores = opt.cider_reward_weight * cider_scores + \
             opt.bleu_reward_weight * bleu_scores + \
             opt.cov_reward_weight * cov_scores + \
             opt.cl_reward_weight * cl_scores

    return scores   # shape: 5n

# 返回S-C分数【每张图片1个得分】
# 返回n个分数
def get_self_cider_scores(data_gts, gen_result, opt):
    """

    :param data_gts: list, (1, 5, d)
    :param gen_result: tensor, (5, d)
    :param opt:
    :return:
    """
    batch_size = gen_result.size(0)  # batch_size = sample_size * seq_per_img # 总句子数 n*5
    seq_per_img = batch_size // len(data_gts)  # 每张图片5个句子

    res = []

    gen_result = gen_result.data.cpu().numpy()
    for i in range(batch_size): # len:
        res.append(array_to_str(gen_result[i]))  # 生成的句子，总共n*5句

    scores = []
    for i in range(len(data_gts)):  # 每张图片，n张图片
        tmp = Cider_scorer.my_self_cider([res[i * seq_per_img:(i + 1) * seq_per_img]])  # 每张图片对应的5个句子，计算Self-CIDEr

        def get_div(eigvals):
            eigvals = np.clip(eigvals, 0, None)
            return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))

        scores.append(get_div(np.linalg.eigvalsh(tmp[0] / 10)))

    scores = np.array(scores)   # shape: n

    return scores  # n张图片对应的Self-Cider分数
