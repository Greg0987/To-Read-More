"""
Preprocess a raw json dataset into hdf5/json files for use in data_loader.py

Input: json file that has the form
[{ file_path: 'path/img.jpg', captions: ['a caption', ...] }, ...]
example element in this list would look like
{'captions': [u'A man with a red helmet on a small moped on a dirt road. ', u'Man riding a motor bike on a dirt road on the countryside.', u'A man riding on the back of a motorcycle.', u'A dirt path with a young person on a motor bike rests to the foreground of a verdant area with a bridge and a background of cloud-wreathed mountains. ', u'A man in a red shirt and a red hat is on a motorcycle on a hill side.'], 'file_path': u'val2014/COCO_val2014_000000391895.jpg', 'id': 391895}

This script reads this json, does some basic preprocessing on the captions
(e.g. lowercase, etc.), creates a special UNK token, and encodes everything to arrays

Output: a json file and an hdf5 file
The hdf5 file contains several fields:
/labels is (M,max_length) uint32 array of encoded labels, zero padded
/label_start_ix and /label_end_ix are (N,) uint32 arrays of pointers to the 
  first and last indices (in range 1..M) of labels for each image
/label_length stores the length of the sequence for each of the M sequences

The json file has a dict that contains:
- an 'ix_to_word' field storing the vocab in form {ix:'word'}, where ix is 1-indexed
- an 'images' field that is a list holding auxiliary information for each image, 
  such as in particular the 'split' it was assigned to.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from pathlib import Path
import json
import argparse
from random import shuffle, seed
import sys
sys.path.append('..')
sys.path.append(os.getcwd())

import string
# non-standard dependencies:
import h5py
import numpy as np
import torch
import torchvision.models as models
# import skimage.io
from PIL import Image

# 返回经过处理加入UNK的词表
def build_vocab(imgs, params):
    count_thr = params['word_count_threshold']  # 超过5才会被放进词表

    # count up the number of words
    counts = {}
    for img in imgs:
        for sent in img['sentences']:
            for w in sent['tokens']:    # w是每个单词
                counts[w] = counts.get(w, 0) + 1
    cw = sorted([(count,w) for w,count in counts.items()], reverse=True)
    print('top words and their counts:')    # 打印出现前20个词
    print('\n'.join(map(str,cw[:20])))

    # print some stats  打印统计信息
    total_words = sum(counts.values())
    print('total words:', total_words)
    bad_words = [w for w,n in counts.items() if n <= count_thr] # 低频词
    vocab = [w for w,n in counts.items() if n > count_thr]      # 剔除低频词
    bad_count = sum(counts[w] for w in bad_words)
    print('number of bad words: %d/%d = %.2f%%' % (len(bad_words), len(counts), len(bad_words)*100.0/len(counts)))
    print('number of words in vocab would be %d' % (len(vocab), ))
    print('number of UNKs: %d/%d = %.2f%%' % (bad_count, total_words, bad_count*100.0/total_words))
    # 未知词UNK即为低频词

    # lets look at the distribution of lengths as well
    sent_lengths = {}
    for img in imgs:
        for sent in img['sentences']:
            txt = sent['tokens']    # 是单词列表
            nw = len(txt)           # 单词数量
            sent_lengths[nw] = sent_lengths.get(nw, 0) + 1  # 记录在词典
    max_len = max(sent_lengths.keys())  # 获取最长句子长度，并打印
    print('max length sentence in raw data: ', max_len)
    print('sentence length distribution (count, number of words):')
    sum_len = sum(sent_lengths.values())    # 总共有多少句子
    for i in range(max_len+1):  # 打印句子长度的分布，包括句子长度和对应句子数量
        print('%2d: %10d   %f%%' % (i, sent_lengths.get(i,0), sent_lengths.get(i,0)*100.0/sum_len))

    # lets now produce the final annotations
    if bad_count > 0:   # 存在未知词，用'UNK'表示
        # additional special UNK token we will use below to map infrequent words to
        print('inserting the special UNK token')
        vocab.append('UNK')
    
    for img in imgs:
        img['final_captions'] = []
        for sent in img['sentences']:
            txt = sent['tokens']    # 是单词列表
            # 将低频词替换为UNK，生成最终注释，并存储于img中
            caption = [w if counts.get(w,0) > count_thr else 'UNK' for w in txt]
            img['final_captions'].append(caption)

    return vocab

# 返回最终经过索引化的描述的数组L，以及开始索引、结束索引、和每个句子的长度数组
def encode_captions(imgs, params, wtoi):
    """ 
    encode all captions into one large array, which will be 1-indexed.
    also produces label_start_ix and label_end_ix which store 1-indexed 
    and inclusive (Lua-style) pointers to the first and last caption for
    each image in the dataset.
    """

    max_length = params['max_length']   # 最大长度
    N = len(imgs)   # 图像数量
    M = sum(len(img['final_captions']) for img in imgs) # total number of captions  描述总数

    label_arrays = []
    label_start_ix = np.zeros(N, dtype='uint32') # note: these will be one-indexed
    label_end_ix = np.zeros(N, dtype='uint32')
    label_length = np.zeros(M, dtype='uint32')
    caption_counter = 0 # 描述计数
    counter = 1
    for i,img in enumerate(imgs):
        n = len(img['final_captions'])  # 每张图片有n个句子
        assert n > 0, 'error: some image has no captions'

        Li = np.zeros((n, max_length), dtype='uint32')  # 零数组用于存储
        for j,s in enumerate(img['final_captions']):    # 遍历每个句子
            # 记录这个句子的长度
            label_length[caption_counter] = min(max_length, len(s)) # record the length of this sequence
            caption_counter += 1
            for k,w in enumerate(s):    # 将句子转化为对应词汇表的索引，并且将长度限制在max_length内
                if k < max_length:
                    Li[j,k] = wtoi[w]

        # note: word indices are 1-indexed, and captions are padded with zeros
        # 词汇表索引从1开始，0是填充
        label_arrays.append(Li) # 索引记录
        label_start_ix[i] = counter         # 每张图片第1句描述对应的索引
        label_end_ix[i] = counter + n - 1   # 每张图片最后1句描述对应的索引
        
        counter += n
    
    L = np.concatenate(label_arrays, axis=0) # put all the labels together  将所有句子连在一起
    # 确保数组的形状和长度正确，并检查是否有注释为空
    assert L.shape[0] == M, 'lengths don\'t match? that\'s weird'
    assert np.all(label_length > 0), 'error: some caption had no words?'

    # 返回最终描述的数组L，以及开始索引、结束索引、和每个句子的长度数组
    # start和end是用于确定每张图在数组中的范围，方便检索
    print('encoded captions to array of size ', L.shape)
    return L, label_start_ix, label_end_ix, label_length

# 写_label.h5，内含所有描述的数组，每张图像对应的开始、结束索引，每个句子的长度数组
# 写.json，新增映射表，图像划分信息，路径和文件名合并，id，图像宽、高等
def main(params):

    imgs = json.load(open(params['input_json'], 'r'))
    imgs = imgs['images']

    seed(123) # make reproducible
    
    # create the vocab  创建词表
    vocab = build_vocab(imgs, params)
    # 创建映射，itow是索引到单词的映射，wtoi是单词到索引的映射
    itow = {i+1:w for i,w in enumerate(vocab)} # a 1-indexed vocab translation table    # 从1开始
    wtoi = {w:i+1 for i,w in enumerate(vocab)} # inverse table
    
    # encode captions in large arrays, ready to ship to hdf5 file
    L, label_start_ix, label_end_ix, label_length = encode_captions(imgs, params, wtoi)

    # create output h5 file
    # 将上面的写入h5文件中：cocotalk_label.h5
    N = len(imgs)
    # f_lb = h5py.File(os.path.join(params['output_dir'], params['output_h5']+'_label.h5'), "w")  # cocotalk_label.h5
    f_lb = h5py.File(params['output_h5']+'_label.h5', "w")  # cocotalk_label.h5
    f_lb.create_dataset("labels", dtype='uint32', data=L)
    f_lb.create_dataset("label_start_ix", dtype='uint32', data=label_start_ix)
    f_lb.create_dataset("label_end_ix", dtype='uint32', data=label_end_ix)
    f_lb.create_dataset("label_length", dtype='uint32', data=label_length)
    f_lb.close()

    # create output json file
    out = {}
    out['ix_to_word'] = itow # encode the (1-indexed) vocab 记录索引到单词的映射表
    out['word_to_ix'] = wtoi
    out['images'] = []  # 用于存储图像信息
    for i,img in enumerate(imgs):
        
        jimg = {}
        jimg['split'] = img['split']    # 获取图像划分信息
        # 如果具有文件名信息，将路径信息和文件名合并，存储到jimg['file_path']中
        if 'filename' in img: jimg['file_path'] = os.path.join(img.get('filepath', ''), img['filename']) # copy it over, might need
        # 根据cocoid或者imgid进行记录
        if 'cocoid' in img:
            jimg['id'] = img['cocoid'] # copy over & mantain an id, if present (e.g. coco ids, useful)
        elif 'imgid' in img:
            jimg['id'] = img['imgid']

        # 图像非空，打开图像记录图像宽高
        if params['images_root'] != '':
            with Image.open(os.path.join(params['images_root'], img['filepath'], img['filename'])) as _img:
                jimg['width'], jimg['height'] = _img.size

        out['images'].append(jimg)  # 将信息记录到字典中
    
    json.dump(out, open(params['output_json'], 'w'))    # 写出信息到['output_json']中
    print('wrote ', params['output_json'])  # 打印输出文件名

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='data/dataset_coco.json', help='input json file to process into hdf5')
    parser.add_argument('--output_json', default='data/cocotalk.json', help='output json file')
    parser.add_argument('--output_h5', default='data/cocotalk', help='output h5 file')
    parser.add_argument('--images_root', default='', help='root location in which images are stored, to be prepended to file_path in input json')

    # options
    parser.add_argument('--max_length', default=16, type=int, help='max length of a caption, in number of words. captions longer than this get clipped.')
    parser.add_argument('--word_count_threshold', default=5, type=int, help='only words that occur more than this number of times will be put in vocab')

    args = parser.parse_args()
    # setattr(args, "save_dir", Path("outputs"))
    params = vars(args) # convert to ordinary dict

    # 将所有的PosixPath类型的值转换为字符串
    params = {k: str(v) if isinstance(v, Path) else v for k, v in params.items()}
    
    print('parsed input parameters:')
    print(json.dumps(params, indent = 2))
    main(params)
