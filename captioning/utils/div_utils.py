from random import uniform
import numpy as np
from collections import OrderedDict, defaultdict
from itertools import tee
import time

# -----------------------------------------------
def find_ngrams(input_list, n):
    return zip(*[input_list[i:] for i in range(n)])

# 返回所有图片的平均Div-n，以及每张图片的Div-n
def compute_div_n(caps,n=1):
  aggr_div = [] # aggregated diversity 多样性总计
  for k in caps:    # 每个key
      all_ngrams = set()
      lenT = 0.
      for c in caps[k]: # 每句caption
         tkns = c.split()   # 每个单词（分词操作）
         lenT += len(tkns)  # caption长度
         ng = find_ngrams(tkns, n)  # n-gram
         all_ngrams.update(ng)      # 每张图片5句话的所有n-gram
      aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT))) # 每张图片的n-gram个数/总单词数
  return np.array(aggr_div).mean(), np.array(aggr_div)

# 全局div-n
def compute_global_div_n(caps,n=1):
  aggr_div = []
  all_ngrams = set()
  lenT = 0.
  for k in caps:
      for c in caps[k]:
         tkns = c.split()
         lenT += len(tkns)
         ng = find_ngrams(tkns, n)
         all_ngrams.update(ng)
  if n == 1:
    aggr_div.append(float(len(all_ngrams)))
  else:
    aggr_div.append(float(len(all_ngrams))/ (1e-6 + float(lenT)))
  return aggr_div[0], np.repeat(np.array(aggr_div),len(caps))