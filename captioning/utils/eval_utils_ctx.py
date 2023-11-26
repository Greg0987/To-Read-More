from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import json
import os
import sys
from . import misc as utils

# load coco-caption if available
# try:
print(os.getcwd())
sys.path.append(os.getcwd())
sys.path.append("./coco_caption")
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
# except:
#     print('Warning: coco-caption not available')

bad_endings = ['a','an','the','in','for','at','of','with','before','after','on','upon','near','to','is','are','am']
bad_endings += ['the']

def count_bad(sen):
    sen = sen.split(' ')
    if sen[-1] in bad_endings:
        return 1
    else:
        return 0

def getCOCO(dataset):
    if 'coco' in dataset:
        annFile = 'coco_caption/annotations/captions_val2014.json'
    elif 'flickr30k' in dataset or 'f30k' in dataset:
        annFile = '../Self-critical.pytorch/data/f30k_captions4eval.json'
    return COCO(annFile)

# 统计准确性和多样性的指标，额外还有坏结尾句子比率，预测的句子等
def language_eval(dataset, preds, preds_n, eval_kwargs, split):
    """

    :param dataset:
    :param preds:   {'image_id': data['infos'][k]['id'], 'caption': sent, 'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
    :param preds_n:
    :param eval_kwargs:
    :param split:
    :return:
    """
    model_id = eval_kwargs['id']
    eval_oracle = eval_kwargs.get('eval_oracle', 0)  # 是否使用oracle
    tags_hdf5 = eval_kwargs.get('tags_hdf5', '')  # 标签的hdf5文件路径
    tags_num = eval_kwargs.get('tags_num', 50)  # 默认计算50个标签

    # create output dictionary  # 创建输出字典
    out = {}

    if len(preds_n)>0:
        # 根据数据集的不同选择不同的json文件路径
        if 'coco' in dataset:
            dataset_file = './data/dataset_coco.json'
        elif 'flickr30k' in dataset or 'f30k' in dataset:
            dataset_file = '../Self-critical.pytorch/data/dataset_flickr30k.json'
        # 获取训练集的句子和生成的句子
        training_sentences = set([' '.join(__['tokens']) for _ in json.load(open(dataset_file))['images'] if
                                  not _['split'] in ['val', 'test'] for __ in _['sentences']])
        generated_sentences = set([_['caption'] for _ in preds_n])

        # ====计算新句子出现率===
        novels = generated_sentences - training_sentences
        out['novel_sentences'] = float(len(novels)) / len(preds_n)
        # 统计unique
        preds_n_dict = {}
        for pred in preds_n:
            img_id = pred['image_id']
            # 将预测的句子加入到字典中，如果字典中没有该图片的id，则创建一个新的set，如果有则将预测的句子加入到set中
            preds_n_dict[img_id] = set(list(list(preds_n_dict[img_id]) + [pred['caption']])) \
                                       if img_id in preds_n_dict else [pred['caption']]
        # 计算unique句子的比率
        out['unique_sentences'] = float(np.sum([len(_) for _ in preds_n_dict.values()])) / len(preds_n)
        print('unique_sentences: ', out['unique_sentences'])
        tmp = [_.split() for _ in generated_sentences]
        words = []
        for _ in tmp:
            words += _
        # 衡量生成句子的词汇量
        out['vocab_size'] = len(words)

    # encoder.FLOAT_REPR = lambda o: format(o, '.3f')
    # 缓存路径
    cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '.json')

    coco = getCOCO(dataset)
    # 获取验证集中所有图片的id
    valids = coco.getImgIds()

    # filter results to only those in MSCOCO validation set (will be about a third)
    # 过滤掉不在验证集中的图片
    preds_filt = [p for p in preds if p['image_id'] in valids]
    # 计算过滤后图片结果的困惑度和信息熵
    mean_perplexity = sum([_['perplexity'] for _ in preds_filt]) / len(preds_filt)
    mean_entropy = sum([_['entropy'] for _ in preds_filt]) / len(preds_filt)
    mean_tags_cov = sum([_['Cov_tags'] for _ in preds_filt]) / len(preds_filt)
    print('using %d/%d predictions' % (len(preds_filt), len(preds)))

    # 记录到JSON文件中
    json.dump(preds_filt, open(cache_path, 'w'))  # serialize to temporary json file. Sigh, COCO API...

    # 用COCO的API计算指标
    cocoRes = coco.loadRes(cache_path)  # 加载结果
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()

    print('computing Cov_tags score...')
    print("tags_cov: %0.3f" % mean_tags_cov)

    # 指标包括Bleu、METEOR、ROUGE_L、CIDEr
    for metric, score in cocoEval.eval.items():
        out[metric] = score
    # Add mean perplexity and mean entropy metrics
    out['perplexity'] = mean_perplexity
    out['entropy'] = mean_entropy
    out['Cov_tags'] = mean_tags_cov

    # SPICE指标需要单独计算
    imgToEval = cocoEval.imgToEval
    for k in list(imgToEval.values())[0]['SPICE'].keys():
        if k != 'All':
            out['SPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEval.values()])
            out['SPICE_'+k] = (out['SPICE_'+k][out['SPICE_'+k] == out['SPICE_'+k]]).mean()
    # 同时将每个图片的caption记录到imgToEval中
    for p in preds_filt:
        image_id, caption = p['image_id'], p['caption']
        imgToEval[image_id]['caption'] = caption


    # 计算多样性指标【当生成多个句子时】
    if len(preds_n)>0:
        from captioning.utils import eval_multi
        # 缓存路径
        cache_path = os.path.join('eval_results/', '.cache_' + model_id + '_' + split + '_n.json')
        # =====ALLSPICE指标=====
        print('====================computing ALLSPICE score====================')
        allspice = eval_multi.eval_allspice(dataset, preds_n, model_id, split)
        out.update(allspice['overall'])
        # =====Div指标和m-BLEU指标=====
        print('====================computing Div and m-BLEU score====================')
        div_stats = eval_multi.eval_div_stats(dataset, preds_n, model_id, split)
        out.update(div_stats['overall'])
        # =====Oracle指标=====
        if eval_oracle:
            print('====================computing Oracle score====================')
            oracle = eval_multi.eval_oracle(dataset, preds_n, model_id, split)
            out.update(oracle['overall'])
        else:
            oracle = None
        # =====Self-CIDEr指标=====
        print('====================computing Self-CIDEr score====================')
        self_cider = eval_multi.eval_self_cider(dataset, preds_n, model_id, split)
        out.update(self_cider['overall'])
        # =====Tags_Cov指标=====
        print('====================computing Tags_Cov score====================')
        tags_cov = eval_multi.eval_tags_coverage(dataset, preds_n, model_id, split, eval_kwargs)
        out.update(tags_cov['overall'])

        # 写到json文件中
        with open(cache_path, 'w') as outfile:
            json.dump({'allspice': allspice, 'div_stats': div_stats,
                       'oracle': oracle, 'self_cider': self_cider,
                       'cov_tags': tags_cov}, outfile)

    # 计算坏结尾句子的比率
    out['bad_count_rate'] = sum([count_bad(_['caption']) for _ in preds_filt]) / float(len(preds_filt))
    outfile_path = os.path.join('eval_results/', model_id + '_' + split + '.json')
    # 写道json文件中
    with open(outfile_path, 'w') as outfile:
        json.dump({'overall':out, 'imgToEval': imgToEval}, outfile)

    return out



def eval_split_n(model, loader, n_predictions, input_data, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)  # 是否打印信息
    beam_size = eval_kwargs.get('beam_size', 1)  # 束搜索大小
    sample_n = eval_kwargs.get('sample_n', 1)  # 生成句子数量
    sample_n_method = eval_kwargs.get('sample_n_method', 'sample')  # 采样方法

    obj_f, grid_f, tag_f, data = input_data
    tmp_eval_kwargs = eval_kwargs.copy()

    # 进行束搜索，采样前1个句子
    if sample_n_method == 'bs':
        tmp_eval_kwargs.update({'sample_n': 1, 'beam_size': sample_n, 'group_size': 1})    # randomness from softmax
        with torch.no_grad():
            model(obj_f, grid_f, tag_f, opt=tmp_eval_kwargs, mode='sample')
        for k in range(obj_f.shape[0]):
            # 获取n个句子
            _sents = utils.decode_sequence(loader.get_vocab(), nn.utils.rnn.pad_sequence(
                [model.done_beams[k][_]['seq'] for _ in range(sample_n)], batch_first=True))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
                n_predictions.append(entry)
    # 随机采样，采样前n个句子
    # sample / gumbel / topk sampling / nucleus sampling
    elif sample_n_method == 'sample' or \
            sample_n_method == 'greedy' or \
            sample_n_method.startswith('top'):
        tmp_eval_kwargs.update({'sample_n': sample_n, 'sample_n_method': sample_n_method, 'beam_size': 1})
        with torch.no_grad():
            _seq, _sampleLogprobs = model(obj_f, grid_f, tag_f, opt=tmp_eval_kwargs, mode='sample')
        # 解码获取句子
        _sents = utils.decode_sequence(loader.get_vocab(), _seq)
        # 计算困惑度
        _perplexity = - _sampleLogprobs.gather(2, _seq.unsqueeze(2)).squeeze(2).sum(1) / ((_seq>0).to(_sampleLogprobs).sum(1) + 1)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                    # 'tags': data['infos'][k // sample_n]['tags'],
                     'perplexity': _perplexity[k].item()}
            n_predictions.append(entry)
    # 多样性束搜索，获取前n个句子
    elif sample_n_method == 'dbs':
        tmp_eval_kwargs.update({'sample_n': sample_n, 'beam_size': beam_size * sample_n, 'group_size': sample_n})
        with torch.no_grad():
            model(obj_f, grid_f, tag_f, opt=tmp_eval_kwargs, mode='sample')
        for k in range(obj_f.shape[0]):
            _sents = utils.decode_sequence(loader.get_vocab(), nn.utils.rnn.pad_sequence(
                [model.done_beams[k][_]['seq'] for _ in range(0, sample_n*beam_size, beam_size)], batch_first=True))
            for sent in _sents:
                entry = {'image_id': data['infos'][k]['id'], 'caption': sent,
                         # 'tags': data['infos'][k]['tags']
                         }
                n_predictions.append(entry)
    else:
        tmp_eval_kwargs.update({'sample_n': sample_n_method[1:], 'group_size': sample_n, 'beam_size':1})
        with torch.no_grad():
            _seq, _sampleLogprobs = model(obj_f, grid_f, tag_f, opt=tmp_eval_kwargs, mode='sample')
        _sents = utils.decode_sequence(loader.get_vocab(), _seq)
        for k, sent in enumerate(_sents):
            entry = {'image_id': data['infos'][k // sample_n]['id'], 'caption': sent,
                     # 'tags': data['infos'][k // sample_n]['tags']
                     }
            n_predictions.append(entry)

    # 进行记录
    if verbose:
        for entry in sorted(n_predictions[-obj_f.shape[0]*sample_n:], key=lambda x: x['image_id']):
            print('image %s: %s' %(entry['image_id'], entry['caption']))


def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 0)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    split = eval_kwargs.get('split', 'val')  # 划分：该数据集是验证集
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)
    sample_n = eval_kwargs.get('sample_n', 1)
    remove_bad_endings = eval_kwargs.get('remove_bad_endings', 0)
    os.environ["REMOVE_BAD_ENDINGS"] = str(
        remove_bad_endings)  # Use this nasty way to make other code clean since it's a global configuration
    device = eval_kwargs.get('device', 'cuda')



    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)  # 重置迭代器

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8   # 防止除以0出错
    predictions = []
    n_predictions = []  # when sample_n > 1; 用于存储多个预测结果
    while True:
        # fetch a batch of data
        data = loader.get_batch(split)  # 获取一个batch的数据
        n = n + len(data['infos'])  # infos是一个list，包含了一个batch的所有图片信息

        obj_f, grid_f, tag_f, labels, masks, tags = \
            [data['obj_f'], data['grid_f'], data['tag_f'], data['labels'], data['masks'],
             data['tags_calc']]
        obj_f = obj_f.cuda()
        grid_f = grid_f.cuda()
        # tag_f = tag_f.cuda()
        tag_f = {
            k1: {
                k2: v2.to(device, non_blocking=True)
                for k2, v2 in v1.items()
            }
            for k1, v1 in tag_f.items()
        }
        tags = tags.cuda()
        labels = labels.cuda()
        masks = masks.cuda()

        if labels is not None and verbose_loss:
            # forward the model to get loss
            with torch.no_grad():
                out = model(obj_f, grid_f, tag_f, labels[..., :-1])
                if isinstance(out, tuple):
                    out = out[0]
                # 按指定的crit计算loss
                loss = crit(out, labels[..., 1:], masks[..., 1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1 # 计数器记录epoch，跟踪模型训练进度

        # forward the model to also get generated samples for each image
        with torch.no_grad():
            tmp_eval_kwargs = eval_kwargs.copy()
            tmp_eval_kwargs.update({'sample_n': 1, 'group_size': 1})
            # print('tmp_eval_kwargs', tmp_eval_kwargs['sample_n'])
            # 随机采样，生成结果
            seq, seq_logprobs = model(obj_f, grid_f, tag_f, opt=tmp_eval_kwargs, mode='sample')
            nn, d = seq.size()  # (b_s, 20)

            # print(seq.size())
            # print(seq_logprobs.size())
            seq = seq.data  # (batch_size, seq_length)
            # 计算交叉熵和困惑度
            entropy = - (F.softmax(seq_logprobs, dim=2) * seq_logprobs).sum(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)
            perplexity = - seq_logprobs.gather(2, seq.unsqueeze(2)).squeeze(2).sum(1) / (
                        (seq > 0).to(seq_logprobs).sum(1) + 1)
            # print(entropy.shape)    # (b_s)
            # print(perplexity.shape) # (b_s)

            # 计算cov
            tags_cov = torch.zeros(nn).to(seq_logprobs)
            for i in range(nn):
                _sentence = [word.item() for word in seq[i] if word != 0]   # 0是填充符号
                # _sentence_set = set(_.item() for _ in seq[i] if _ != 0]
                tags_set = set(tag.item() for tag in tags[i])
                # print('=' * 30)
                # print(_sentence)
                # print(tags_set)
                common_words_count = [_ for _ in _sentence if _ in tags_set]
                # print(_sentence, '\n', tags_set, '\n', common_words_count)

                # print(common_words_count)
                # print(len(_sentence))

                tags_cov[i] = len(common_words_count) / len(_sentence) if len(_sentence) > 0 else 0
                # print(tags_cov[i].item())
            # print(tags_cov)

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(obj_f.shape[0]):
                print('\n'.join(
                    [utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        # 将序列解码成句子
        sents = utils.decode_sequence(loader.get_vocab(), seq)
        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent, 'Cov_tags': tags_cov[k].item(),
                     'perplexity': perplexity[k].item(), 'entropy': entropy[k].item()}
            if eval_kwargs.get('dump_path', 0) == 1:    # 用于指定是否将生成结果保存为文件
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:  # 用于指定是否将输入的图片保存
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

            if verbose: # 打印详细信息
                print('image %s: %s' %(entry['image_id'], entry['caption']))

        if sample_n > 1:    # 多次采样，将多个句子添加到n_predictions中
            eval_split_n(model, loader, n_predictions, [obj_f, grid_f, tag_f, data], eval_kwargs)

        ix1 = data['bounds']['it_max']  # 该batch的最大迭代次数
        if num_images != -1:    # 对图像数进行限制
            ix1 = min(ix1, num_images)
        else:
            num_images = ix1
        for i in range(n-ix1):
            predictions.pop()   # 把多余元素pop，确保其中元素与数据集中的图像数量相同

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(n, ix1, loss))

        if num_images >= 0 and n >= num_images:
            break

    lang_stats = None

    if len(n_predictions) > 0 and 'perplexity' in n_predictions[0]:
        n_predictions = sorted(n_predictions, key=lambda x: x['perplexity'])  # 按困惑度排序
    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')
    torch.save((predictions, n_predictions), os.path.join('eval_results/', '.saved_pred_'+eval_kwargs['id']+'_'+split+'.pth')) # 保存预测结果

    if lang_eval == 1:
        lang_stats = language_eval(dataset, predictions, n_predictions, eval_kwargs, split)

    # Switch back to training mode
    model.train()
    # 返回每个epoch的val_loss，预测句子，语言指标
    return loss_sum/loss_evals, predictions, n_predictions, lang_stats



