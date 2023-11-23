from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import h5py
from lmdbdict import lmdbdict
from lmdbdict.methods import DUMPS_FUNC, LOADS_FUNC
import os
import numpy as np
import numpy.random as npr
import random
from functools import partial

import pdb
import inspect

import torch
import torch.utils.data as data

import multiprocessing
import six

# get()方法返回获取到的特征数据
class HybridLoader: # 混合数据源加载器
    """
    If db_path is a director, then use normal file loading
    If lmdb, then load from lmdb
    The loading method depend on extention.

    in_memory: if in_memory is True, we save all the features in memory
               For individual np(y|z)s, we don't need to do that because the system will do this for us.
               Should be useful for lmdb or h5.
               (Copied this idea from vilbert)
    """
    def __init__(self, db_path, ext, in_memory=False):
        self.db_path = db_path  # 数据源路径
        self.ext = ext          # 文件扩展名
        if self.ext == '.npy':
            self.loader = lambda x: np.load(six.BytesIO(x))
        else:
            def load_npz(x):
                x = np.load(six.BytesIO(x))
                return x['feat'] if 'feat' in x else x['z']  # normally it should be 'feat', but under cocotest_bu, the key is saved to be 'z' mistakenly.
            self.loader = load_npz  # 加载数据

        # 根据路径结尾，用不同的数据源类型进行处理
        if db_path.endswith('.lmdb'):
            self.db_type = 'lmdb'
            self.lmdb = lmdbdict(db_path, unsafe=True)
            self.lmdb._key_dumps = DUMPS_FUNC['ascii']
            self.lmdb._value_loads = LOADS_FUNC['identity']
        elif db_path.endswith('.pth'): # Assume a key,value dictionary
            self.db_type = 'pth'
            self.feat_file = torch.load(db_path, encoding='bytes')
            self.loader = lambda x: x
            print('HybridLoader: ext is ignored')
        elif db_path.endswith('h5') or db_path.endswith('hdf5'):    # add by Greg when to be used in DiM2T
            self.db_type = 'h5'
            # self.loader = lambda x: np.array(x).astype('float32')
            self.loader = lambda x: x
        else:
            self.db_type = 'dir'

        self.in_memory = in_memory
        if self.in_memory:
            self.features = {}
    
    def get(self, key):

        if self.in_memory and key in self.features:
            # We save f_input because we want to save the
            # compressed bytes to save memory
            f_input = self.features[key]
        elif self.db_type == 'lmdb':
            f_input = self.lmdb[key]
        elif self.db_type == 'pth':
            f_input = self.feat_file[key.encode()]
        elif self.db_type == 'h5':
            f_input = h5py.File(self.db_path, 'r')[key] # 使用键获取特征数据
        else:
            # 普通文件通过拼接路径，以二进制读取方式打开读取内容
            f_input = open(os.path.join(self.db_path, key + self.ext), 'rb').read()

        if self.in_memory and key not in self.features: # 将特征数据存储到内存字典中
            self.features[key] = f_input

        # load image
        feat = self.loader(f_input) # 加载图片？

        return feat

class Dataset(data.Dataset):
    # 获取词汇表大小
    def get_vocab_size(self):
        return self.vocab_size
    # 获取词汇表
    def get_vocab(self):
        return self.ix_to_word
    # 映射成token
    def to_tokens(self):
        return self.word_to_ix
    # 获取序列长度
    def get_seq_length(self):
        return self.seq_length

    def __init__(self, opt):
        self.opt = opt
        self.seq_per_img = opt.seq_per_img  # 每张图片的句子数
        
        # feature related options   # 相关配置
        self.use_fc = getattr(opt, 'use_fc', True)
        self.use_att = getattr(opt, 'use_att', True)
        self.use_box = getattr(opt, 'use_box', 0)
        self.norm_att_feat = getattr(opt, 'norm_att_feat', 0)
        self.norm_box_feat = getattr(opt, 'norm_box_feat', 0)

        # add by Greg to use New features to Projector
        self.new_features = getattr(opt, 'new_features', True)
        self.max_detections = getattr(opt, 'max_detections', 50)
        self.k = getattr(opt, 'topk', 9)   # The nums of tags to use
        self.tags_num = getattr(opt, 'tags_num', 100) # The nums of tags to calculate the ov

        # load the json file which contains additional information about the dataset
        # 加载包含数据集附加信息的JSON文件，并提取词汇表相关信息
        print('DataLoader loading json file: ', opt.input_json) # 打印正在加载的json文件路径
        self.info = json.load(open(self.opt.input_json))    # 解析cocotalk.json
        if 'ix_to_word' in self.info:
            self.ix_to_word = self.info['ix_to_word']   # 词汇表映射
            self.vocab_size = len(self.ix_to_word)      # 词汇表大小
            print('vocab size is ', self.vocab_size)    # 打印
        if 'word_to_ix' in self.info:
            self.word_to_ix = self.info['word_to_ix']   # 词汇表映射

        # open the hdf5 file

        """
        Setting input_label_h5 to none is used when only doing generation.
        For example, when you need to test on coco test set.
        只有在做生成的时候，设置input_label_h5为None，例如需要在COCO测试集上进行test
        """
        if self.opt.input_label_h5 != '':
            # 导入cocotalk_label.h5
            self.h5_label_file = h5py.File(self.opt.input_label_h5, 'r', driver='core') # 只读模式打开h5文件
            # load in the sequence data 记载序列数据
            seq_size = self.h5_label_file['labels'].shape   # 加载序列形状
            self.label = self.h5_label_file['labels'][:]    # 标签    # 里面都是tokens
            self.seq_length = seq_size[1]                   # 序列长度
            print('max sequence length in data is', self.seq_length)    # 打印序列最大长度
            # load the pointers in full to RAM (should be small enough)
            self.label_start_ix = self.h5_label_file['label_start_ix'][:]   # 起始索引
            self.label_end_ix = self.h5_label_file['label_end_ix'][:]       # 结束索引
        else:
            self.seq_length = 1 # 否则将序列长度设置为1，表示每个样本只有1个标签，通常用于生成阶段

        self.data_in_memory = getattr(opt, 'data_in_memory', False)
        if self.new_features:
            print('DataLoader loading h5 file: ', opt.obj_dir, opt.grid_dir, opt.tag_dir,
                  opt.input_label_h5)
            self.obj_loader = HybridLoader(self.opt.obj_dir, '.h5', in_memory=self.data_in_memory)
            self.grid_loader = HybridLoader(self.opt.grid_dir, '.h5', in_memory=self.data_in_memory)
            self.tag_loader = HybridLoader(self.opt.tag_dir, '.h5', in_memory=self.data_in_memory)
            self.tags_to_calc = HybridLoader(self.opt.tags_h5, '.h5', in_memory=self.data_in_memory)
        else:
            print('DataLoader loading h5 file: ', opt.input_fc_dir, opt.input_att_dir, opt.input_box_dir,
                  opt.input_label_h5)
            self.fc_loader = HybridLoader(self.opt.input_fc_dir, '.npy', in_memory=self.data_in_memory) # 加载图像特征
            self.att_loader = HybridLoader(self.opt.input_att_dir, '.npz', in_memory=self.data_in_memory)   # 加载图像区域特征
            self.box_loader = HybridLoader(self.opt.input_box_dir, '.npy', in_memory=self.data_in_memory)   # 加载图像边界框特征
            self.tag_loader = HybridLoader(self.opt.tag_dir, '.h5', in_memory=self.data_in_memory)
            self.tags_to_calc = HybridLoader(self.opt.tags_h5, '.h5', in_memory=self.data_in_memory)

        self.num_images = len(self.info['images']) # self.label_start_ix.shape[0]   #获取图像数量
        print('read %d image features' %(self.num_images))  # 打印读取的图像数量

        # separate out indexes for each of the provided splits
        # 将图像，按照不同划分，分配到不同的列表中
        self.split_ix = {'train': [], 'val': [], 'test': []}    # 训练集，验证集，测试集
        for ix in range(len(self.info['images'])):
            img = self.info['images'][ix]   # 获取图像信息
            if not 'split' in img:  # 不存在'split'键，则表示没有任何划分，那么都进行划分
                self.split_ix['train'].append(ix)
                self.split_ix['val'].append(ix)
                self.split_ix['test'].append(ix)
            elif img['split'] == 'train':
                self.split_ix['train'].append(ix)
            elif img['split'] == 'val':
                self.split_ix['val'].append(ix)
            elif img['split'] == 'test':
                if opt.train_only == 0:
                    self.split_ix['test'].append(ix)
                else:
                    self.split_ix['val'].append(ix)
            elif opt.train_only == 0: # restval # 同时用于test和val
                self.split_ix['train'].append(ix)
            else:
                self.split_ix['val'].append(ix)

        # 此处ix是cocotalk.json当中的索引，从0开始，到len("images")为止
        print('assigned %d images to split train' %len(self.split_ix['train']))
        print('assigned %d images to split val' %len(self.split_ix['val']))
        print('assigned %d images to split test' %len(self.split_ix['test']))

    # 按照ix返回seq_per_img条序列标注
    def get_captions(self, ix, seq_per_img):
        # fetch the sequence labels
        # 获取序列表标签
        # 获取起始和结束索引
        ix1 = self.label_start_ix[ix] - 1 #label_start_ix starts from 1
        ix2 = self.label_end_ix[ix] - 1
        ncap = ix2 - ix1 + 1 # number of captions available for this image 可用于该张图片的句子数量
        assert ncap > 0, 'an image does not have any label. this can be handled but right now isn\'t'

        # 若实际数目不足5条，需要进行有放回的采样补足
        if ncap < seq_per_img:
            # we need to subsample (with replacement)
            seq = np.zeros([seq_per_img, self.seq_length], dtype = 'int')
            for q in range(seq_per_img):
                ixl = random.randint(ix1,ix2)
                seq[q, :] = self.label[ixl, :self.seq_length]
        # 超过5条则随机选连续5条
        else:
            ixl = random.randint(ix1, ix2 - seq_per_img + 1)
            seq = self.label[ixl: ixl + seq_per_img, :self.seq_length]

        return seq

    def collate_func(self, batch, split):   # 对一个batch的样本进行处理和组合
        seq_per_img = self.seq_per_img
        # print("开始继续进行打包batch")
        # pdb.set_trace()
        if self.new_features:
            obj_batch = []
            grid_1_batch = []
            grid_4_batch = []
            grid_9_batch = []
            grid_16_batch = []
            grid_25_batch = []
            # tag_f_batch = []
            # tag_t_batch = []
            tag_1_batch = []
            tag_4_batch = []
            tag_9_batch = []
            tag_16_batch = []
            tag_25_batch = []
            tags_calc_batch = []    # used for tags cov cal
            label_batch = []

            wrapped = False

            infos = []
            gts = []

            for sample in batch:
                tmp_obj, tmp_grid, tmp_tag, tmp_tags_calc, tmp_seq,\
                ix, it_pos_now, tmp_wrapped = sample
                if tmp_wrapped:
                    wrapped = True

                grid_1_, grid_4_, grid_9_, grid_16_, grid_25_ = tmp_grid
                grid_1_batch.append(grid_1_)
                grid_4_batch.append(grid_4_)
                grid_9_batch.append(grid_9_)
                grid_16_batch.append(grid_16_)
                grid_25_batch.append(grid_25_)

                # version_v2
                # tag_f_, tag_t_ = tmp_tag
                # tag_f_batch.append(tag_f_)
                # tag_t_batch.append(tag_t_)
                # tag_token_batch.append([self.word_to_ix[t.decode()]
                #                         if t.decode() in self.word_to_ix
                #                         else self.word_to_ix['UNK'] for t in tag_t_])

                tag_1_f_, tag_4_f_, tag_9_f_, tag_16_f_, tag_25_f_ = tmp_tag
                # tag_1_t_, tag_4_t_, tag_9_t_, tag_16_t_, tag_25_t_ = tmp_tag
                tag_1_batch.append(tag_1_f_)
                tag_4_batch.append(tag_4_f_)
                tag_9_batch.append(tag_9_f_)
                tag_16_batch.append(tag_16_f_)
                tag_25_batch.append(tag_25_f_)
                # tag_ = np.unique(np.concatenate((tag_1_t_.flatten(), tag_4_t_.flatten(),
                #                                  tag_9_t_.flatten(), tag_16_t_.flatten(),
                #                                 tag_25_t_.flatten()))).tolist()
                # tag_ = [self.word_to_ix[t.decode()]
                #                        if t.decode() in self.word_to_ix
                #                         else self.word_to_ix['UNK'] for t in tag_]  # 将tag转换为ix
                # tag_ = tag_ + [0] * ((1+4+9+16+25)*self.k - len(tag_))    # 用0补齐
                # tag_token_batch.append(tag_)


                # 用于奖励
                tags_calc_batch.append(tmp_tags_calc)

                obj_batch.append(tmp_obj)

                tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype='int')  # 零数组
                if hasattr(self, 'h5_label_file'):
                    # if there is ground truth  如果有gt
                    tmp_label[:, 1: self.seq_length + 1] = tmp_seq  # 复制到第1位和倒数第2位（留出bos和eos）
                label_batch.append(tmp_label)  # 将gt添加到列表中


                # Used for reward evaluation    # 用于奖励评价
                if hasattr(self, 'h5_label_file'):
                    # if there is ground truth  有gt则从label数组中提取出来，存到gts列表中
                    tmp = self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]]
                    if len(tmp) < seq_per_img:
                        # we need to subsample (with replacement)
                        _tmp = np.zeros([seq_per_img, self.seq_length], dtype='int')
                        # print(_tmp)
                        for q in range(seq_per_img):
                            ixl = random.randint(0, len(tmp)-1)
                            _tmp[q, :] = tmp[ixl, :self.seq_length]
                    elif len(tmp) == seq_per_img:
                        _tmp = tmp[:, :self.seq_length]
                    # 超过5条则随机选连续5条
                    else:
                        ixl = random.randint(0, len(tmp) - seq_per_img - 1)
                        _tmp = tmp[ixl: ixl + seq_per_img, :self.seq_length]

                    if len(_tmp)!= 5: print(len(_tmp), _tmp, "\n", seq_per_img)
                    assert len(_tmp) == 5, 'error! tmp length is not 5'

                    gts.append(_tmp)
                else:
                    gts.append([])  # 否则加入空列表

                # record associated info as well    记录样本相关信息
                info_dict = {}
                info_dict['ix'] = ix
                info_dict['id'] = self.info['images'][ix]['id']
                info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
                # info_dict['tags'] = tag_t_  # 此处tags是文本列表
                infos.append(info_dict)  # 将字典进行存储


            # 记录data字典中进行存储
            data = {}
            data['obj_f'] = torch.from_numpy(np.stack(obj_batch))
            data['grid_f'] = {'whole': torch.from_numpy(np.stack(grid_1_batch)),
                            'four': torch.from_numpy(np.stack(grid_4_batch)),
                            'nine': torch.from_numpy(np.stack(grid_9_batch)),
                            'sixteen': torch.from_numpy(np.stack(grid_16_batch)),
                            'twentyfive': torch.from_numpy(np.stack(grid_25_batch))}
            # data['tag_f'] = {'features': torch.from_numpy(np.stack(tag_f_batch)),
            #                'tags': np.stack(tag_token_batch)}
            data['tag_f'] = {'whole': torch.from_numpy(np.stack(tag_1_batch)),
                             'four': torch.from_numpy(np.stack(tag_4_batch)),
                            'nine': torch.from_numpy(np.stack(tag_9_batch)),
                            'sixteen': torch.from_numpy(np.stack(tag_16_batch)),
                            'twentyfive': torch.from_numpy(np.stack(tag_25_batch))}

            data['tags_calc'] = torch.from_numpy(np.stack(tags_calc_batch))
            # 垂直堆叠标注
            data['labels'] = np.vstack(label_batch)
            # generate mask
            # 先计算非0个数+2（bos和eos）

            nonzeros = np.array(list(map(lambda x: (x != 0).sum() + 2, data['labels'])))
            # 生成掩码，非0位置置为1
            mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype='float32')
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            data['masks'] = mask_batch


            data['labels'] = torch.from_numpy(data['labels'].reshape(len(batch), seq_per_img, -1))  # (b_s, seq_per_img, d)
            data['masks'] = torch.from_numpy(data['masks'].reshape(len(batch), seq_per_img, -1))  # 同上

            data['gts'] = gts
            data['bounds'] = {'it_pos_now': it_pos_now,
                              'it_max': len(self.split_ix[split]),
                              'wrapped': wrapped}
            data['infos'] = infos

            # 将所有ndarray转换为torch张量，然后返回data
            # data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
            #        data.items()}  # Turn all ndarray to torch tensor
            return data

        else:
            fc_batch = []
            att_batch = []
            tags_calc_batch = []  # used for tags cov cal
            label_batch = []

            wrapped = False

            infos = []
            gts = []

            for sample in batch:    # 遍历batch中的每个样本
                # fetch image
                tmp_fc, tmp_att, tmp_tags_calc, tmp_seq, \
                    ix, it_pos_now, tmp_wrapped = sample
                if tmp_wrapped: # 如果过程发生了截断或填充，将标记置为True
                    wrapped = True

                fc_batch.append(tmp_fc) # 记录fc和att
                att_batch.append(tmp_att)

                tmp_label = np.zeros([seq_per_img, self.seq_length + 2], dtype = 'int') # 零数组
                if hasattr(self, 'h5_label_file'):
                    # if there is ground truth  如果有gt
                    tmp_label[:, 1 : self.seq_length + 1] = tmp_seq # 复制到第1位和倒数第2位（留出bos和eos）
                label_batch.append(tmp_label)   # 将gt添加到列表中
                # 用于奖励
                tags_calc_batch.append(tmp_tags_calc)


                # Used for reward evaluation    # 用于奖励评价
                if hasattr(self, 'h5_label_file'):
                    # if there is ground truth  有gt则从label数组中提取出来，存到gts列表中
                    gts.append(self.label[self.label_start_ix[ix] - 1: self.label_end_ix[ix]])
                else:
                    gts.append([])  # 否则加入空列表

                # record associated info as well    记录样本相关信息
                info_dict = {}
                info_dict['ix'] = ix
                info_dict['id'] = self.info['images'][ix]['id']
                info_dict['file_path'] = self.info['images'][ix].get('file_path', '')
                infos.append(info_dict) # 将字典进行存储

            # #sort by att_feat length
            # fc_batch, att_batch, label_batch, gts, infos = \
            #     zip(*sorted(zip(fc_batch, att_batch, np.vsplit(label_batch, batch_size), gts, infos), key=lambda x: len(x[1]), reverse=True))
            # 根据长度进行排序，确保最长样本在第一个位置，便于后续处理进行填充
            # 解压操作
            fc_batch, att_batch, tags_calc_batch, label_batch, gts, infos = \
                zip(*sorted(zip(fc_batch, att_batch, tags_calc_batch, label_batch, gts, infos), key=lambda x: 0, reverse=True))
            data = {}
            data['fc_feats'] = np.stack(fc_batch)
            # merge att_feats
            max_att_len = max([_.shape[0] for _ in att_batch])  # 获取注意力最大长度
            # 全零数组
            data['att_feats'] = np.zeros([len(att_batch), max_att_len, att_batch[0].shape[1]], dtype = 'float32')
            # 挨个复制注意力，在行尾用0掩码
            for i in range(len(att_batch)):
                data['att_feats'][i, :att_batch[i].shape[0]] = att_batch[i]
            data['att_masks'] = np.zeros(data['att_feats'].shape[:2], dtype='float32')
            for i in range(len(att_batch)):
                data['att_masks'][i, :att_batch[i].shape[0]] = 1
            # set att_masks to None if attention features have same length
            # 如果所有注意力特征的长度都相同，则其设为None
            if data['att_masks'].sum() == data['att_masks'].size:
                data['att_masks'] = None

            # 垂直堆叠标签
            data['labels'] = np.vstack(label_batch)
            # generate mask
            # 先计算非0个数+2（bos和eos）
            nonzeros = np.array(list(map(lambda x: (x != 0).sum()+2, data['labels'])))
            # 生成掩码，非0位置置为1
            mask_batch = np.zeros([data['labels'].shape[0], self.seq_length + 2], dtype = 'float32')
            for ix, row in enumerate(mask_batch):
                row[:nonzeros[ix]] = 1
            data['masks'] = mask_batch
            data['labels'] = data['labels'].reshape(len(batch), seq_per_img, -1)    # (b_s, seq_per_img, d)
            data['masks'] = data['masks'].reshape(len(batch), seq_per_img, -1)      # 同上

            data['tags_calc'] = np.stack(tags_calc_batch)
            data['gts'] = gts # all ground truth captions of each images
            """
            it_pos_now: 最后一个样本的位置指示符，用于记录当前处理的位置
            it_max: 表示数据集某个split划分的最大位置值，即数据集中样本总数
            wrapped: 表示数据集是否已经遍历完一次
            当it_pos_now到达it_max时，wrapped被设置为True，指示数据集已经完整遍历一次
            """
            data['bounds'] = {'it_pos_now': it_pos_now, # the it_pos_now of the last sample
                              'it_max': len(self.split_ix[split]), 'wrapped': wrapped}
            data['infos'] = infos
            # 将所有ndarray转换为torch张量，然后返回data
            data = {k:torch.from_numpy(v) if type(v) is np.ndarray else v for k,v in data.items()} # Turn all ndarray to torch tensor

            return data

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        返回元组，进一步传递给collate_fn
        """
        ix, it_pos_now, wrapped = index #self.split_ix[index]
        # 获取图像特征
        # str(self.info['images'][ix]['id'])是图像id

        if self.new_features:
            obj_feat = self.obj_loader.get(str(self.info['images'][ix]['id']))
            # obj_n = obj_feat["num_boxes"][()]
            obj = obj_feat["obj_features"][:]   # (n, d) 此处n为50
            n, d = obj.shape

            delta = self.max_detections - n
            if delta > 0:
                p = np.zeros((delta, d), dtype=obj.dtype)
                obj = np.concatenate([obj, p], axis=0)
            elif delta < 0:
                obj = obj[:self.max_detections]

            grid_feat = self.grid_loader.get(str(self.info['images'][ix]['id']))
            grid_1 = grid_feat["whole"][:]
            grid_4 = grid_feat["four"][:]
            grid_9 = grid_feat["nine"][:]
            grid_16 = grid_feat["sixteen"][:]
            grid_25 = grid_feat["twentyfive"][:]
            grid = (grid_1, grid_4, grid_9, grid_16, grid_25)

            # version_v2_tags
            # 可考虑是否用全0张量和UNK进行填充
            # tag_feat = self.tag_loader.get(str(self.info['images'][ix]['id']))
            # tag_f = tag_feat["features"][:self.k]   # 选取前k个
            # tag_t = tag_feat["tags"][:self.k]
            # tag = (tag_f, tag_t)

            tag_feat = self.tag_loader.get(str(self.info['images'][ix]['id']))

            tag_1_f = tag_feat["whole"]["features"][:self.k]   # 选取前k个
            # tag_1_t = tag_feat["whole"]["tags"][:self.k]
            tag_4_f = tag_feat["four"]["features"][:,:self.k,:]
            # tag_4_t = tag_feat["four"]["tags"][:,:self.k]
            tag_9_f = tag_feat["nine"]["features"][:,:self.k,:]
            # tag_9_t = tag_feat["nine"]["tags"][:,:self.k]
            tag_16_f = tag_feat["sixteen"]["features"][:,:self.k,:]
            # tag_16_t = tag_feat["sixteen"]["tags"][:,:self.k]
            tag_25_f = tag_feat["twentyfive"]["features"][:,:self.k,:]
            # tag_25_t = tag_feat["twentyfive"]["tags"][:,:self.k]

            tag_1_f = tag_1_f.reshape(self.k, -1)
            tag_4_f = tag_4_f.reshape((4*self.k, -1))
            tag_9_f = tag_9_f.reshape((9*self.k, -1))
            tag_16_f = tag_16_f.reshape((16*self.k, -1))
            tag_25_f = tag_25_f.reshape((25*self.k, -1))
            tag = (tag_1_f, tag_4_f, tag_9_f, tag_16_f, tag_25_f)
                   # tag_1_t, tag_4_t, tag_9_t, tag_16_t, tag_25_t)

            # 用于计算标签覆盖率
            tags_to_calc = self.tags_to_calc.get(str(self.info['images'][ix]['id']))
            tags_to_calc = tags_to_calc['id'][:]
            n = tags_to_calc.shape[0]
            delta = self.tags_num - n
            if delta > 0:
                p = np.zeros((delta), dtype=tags_to_calc.dtype)
                tags_to_calc = np.concatenate([tags_to_calc, p], axis=0)
            elif delta <= 0:
                tags_to_calc = tags_to_calc[:self.tags_num]

            if hasattr(self, 'h5_label_file'):
                seq = self.get_captions(ix, self.seq_per_img)   # 获取标注序列
            else:
                seq = None

            return [obj, grid, tag, tags_to_calc, seq,
                    ix, it_pos_now, wrapped]

        else:
            if self.use_att:
                att_feat = self.att_loader.get(str(self.info['images'][ix]['id']))
                # Reshape to K x C
                att_feat = att_feat.reshape(-1, att_feat.shape[-1])
                if self.norm_att_feat:
                    att_feat = att_feat / np.linalg.norm(att_feat, 2, 1, keepdims=True)
                if self.use_box:
                    box_feat = self.box_loader.get(str(self.info['images'][ix]['id']))
                    # devided by image width and height
                    x1,y1,x2,y2 = np.hsplit(box_feat, 4)
                    h,w = self.info['images'][ix]['height'], self.info['images'][ix]['width']
                    box_feat = np.hstack((x1/w, y1/h, x2/w, y2/h, (x2-x1)*(y2-y1)/(w*h))) # question? x2-x1+1??
                    if self.norm_box_feat:
                        box_feat = box_feat / np.linalg.norm(box_feat, 2, 1, keepdims=True)
                    att_feat = np.hstack([att_feat, box_feat])
                    # sort the features by the size of boxes
                    # 根据边界框大小进行排序
                    att_feat = np.stack(sorted(att_feat, key=lambda x:x[-1], reverse=True))
            else:
                att_feat = np.zeros((0,0), dtype='float32')

            # 获取全连接特征
            if self.use_fc:
                try:
                    fc_feat = self.fc_loader.get(str(self.info['images'][ix]['id']))
                except:
                    # Use average of attention when there is no fc provided (For bottomup feature)
                    # 当没有提供全连接特征时，使用注意力特征的平均值（使用于bottom-up特征）
                    fc_feat = att_feat.mean(0)
            else:
                fc_feat = np.zeros((0), dtype='float32')

            # 用于计算标签覆盖率
            tags_to_calc = self.tags_to_calc.get(str(self.info['images'][ix]['id']))
            tags_to_calc = tags_to_calc['id'][:]
            n = tags_to_calc.shape[0]
            delta = self.tags_num - n
            if delta > 0:
                p = np.zeros((delta), dtype=tags_to_calc.dtype)
                tags_to_calc = np.concatenate([tags_to_calc, p], axis=0)
            elif delta <= 0:
                tags_to_calc = tags_to_calc[:self.tags_num]

            if hasattr(self, 'h5_label_file'):
                seq = self.get_captions(ix, self.seq_per_img)   # 获取标注序列
            else:
                seq = None
            # 返回一个元组，包含全连接特征、注意力特征、标注序列以及图像索引、当前位置、是否回环标志等信息
            return (fc_feat,
                    att_feat, tags_to_calc, seq,
                    ix, it_pos_now, wrapped)

    # 返回图像张数
    def __len__(self):
        return len(self.info['images'])

class DataLoader:   # 用于加载数据集并生成批次数据
    def __init__(self, opt):
        self.opt = opt
        self.batch_size = self.opt.batch_size
        self.dataset = Dataset(opt)

        # Initialize loaders and iters  初始化加载器和迭代器
        self.loaders, self.iters = {}, {}
        for split in ['train', 'val', 'test']:
            if split == 'train':
                # 对于训练集，使用自定义的采样器并打乱数据
                sampler = MySampler(self.dataset.split_ix[split], shuffle=True, wrap=True)
            else:
                # 对于验证集和测试集，不打乱数据
                sampler = MySampler(self.dataset.split_ix[split], shuffle=False, wrap=False)
            # 创建对应的数据加载器
            self.loaders[split] = data.DataLoader(dataset=self.dataset,
                                                  batch_size=self.batch_size,
                                                  sampler=sampler,
                                                  pin_memory=getattr(opt, 'pin_memory', False),
                                                  num_workers=getattr(opt, 'num_workers', 4), # 4 is usually enough
                                                  collate_fn=partial(self.dataset.collate_func, split=split),
                                                  drop_last=False)
            # 创建对应的迭代器
            self.iters[split] = iter(self.loaders[split])

    # 获取下一个批次的数据
    def get_batch(self, split):
        try:# 尝试获取下一个批次数据
            data = next(self.iters[split])
            # print('dataloader里的data: ', data[0])
        except StopIteration:
            # 如果迭代器已经结束，则重新创建迭代器并获取数据
            self.iters[split] = iter(self.loaders[split])
            data = next(self.iters[split])
        return data

    # 重置迭代器，重新开始获取指定数据集（split)的数据
    def reset_iterator(self, split):
        self.loaders[split].sampler._reset_iter()
        self.iters[split] = iter(self.loaders[split])
    # 获取词汇表大小
    def get_vocab_size(self):
        return self.dataset.get_vocab_size()

    @property
    def vocab_size(self):
        return self.get_vocab_size()

    def get_vocab(self):
        return self.dataset.get_vocab()

    def get_seq_length(self):
        return self.dataset.get_seq_length()

    @property
    def seq_length(self):
        return self.get_seq_length()
    # 返回数据加载器的状态字典，包括各个数据集划分的采样器的状态
    def state_dict(self):
        def get_prefetch_num(split):
            if self.loaders[split].num_workers > 0:
                return (self.iters[split]._send_idx - self.iters[split]._rcvd_idx) * self.batch_size
            else:
                return 0
        return {split: loader.sampler.state_dict(get_prefetch_num(split)) \
                    for split, loader in self.loaders.items()}

    # 加载数据加载器的状态字典，用于恢复数据加载器的状态
    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        for split in self.loaders.keys():
            self.loaders[split].sampler.load_state_dict(state_dict[split])

class MySampler(data.sampler.Sampler):
    def __init__(self, index_list, shuffle, wrap):
        self.index_list = index_list
        self.shuffle = shuffle
        self.wrap = wrap
        # if wrap, there will be not stop iteration called
        # wrap True used during training, and wrap False used during test.
        self._reset_iter()

    def __iter__(self):
        return self

    # 定义了迭代逻辑，返回索引元组，包括当前索引、迭代计数器和是否循环包装的标志
    def __next__(self):
        wrapped = False
        if self.iter_counter == len(self._index_list):
            self._reset_iter()
            if self.wrap:
                wrapped = True
            else:
                raise StopIteration()
        if len(self._index_list) == 0: # overflow when 0 samples
            return None
        elem = (self._index_list[self.iter_counter], self.iter_counter+1, wrapped)
        self.iter_counter += 1
        return elem

    # 用于向后兼容，返回下一个元素
    def next(self):
        return self.__next__()

    # 重置迭代器，根据洗牌选项重新排列索引列表，并将迭代计数器重置为0
    def _reset_iter(self):
        if self.shuffle:
            rand_perm = npr.permutation(len(self.index_list))
            self._index_list = [self.index_list[_] for _ in rand_perm]
        else:
            self._index_list = self.index_list

        self.iter_counter = 0

    # 返回索引列表的长度，用于确定采样器的长度。
    def __len__(self):
        return len(self.index_list)

    def load_state_dict(self, state_dict=None):
        if state_dict is None:
            return
        self._index_list = state_dict['index_list']
        self.iter_counter = state_dict['iter_counter']

    def state_dict(self, prefetched_num=None):
        prefetched_num = prefetched_num or 0
        return {
            'index_list': self._index_list,
            'iter_counter': self.iter_counter - prefetched_num
        }

    