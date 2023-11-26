from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import os
from captioning.utils.eval_utils import getCOCO
import h5py

from .div_utils import compute_div_n, compute_global_div_n

import sys
import os
try:
    sys.path.append("./coco_caption")
    annFile = 'annotations/captions_val2014.json'
    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap
    from pycocoevalcap.eval_spice import COCOEvalCapSpice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.bleu.bleu import Bleu
    sys.path.append("./cider")
    from pyciderevalcap.cider.cider import Cider
    sys.path.append(os.getcwd())

except:
    print('Warning: requirements for eval_multi not satisfied')


def eval_allspice(dataset, preds_n, model_id, split):
    coco = getCOCO(dataset)
    valids = coco.getImgIds()
    
    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    # filter results to only those in MSCOCO validation set (will be about a third)
    preds_filt_n = [p for p in preds_n if p['image_id'] in valids]
    print('using %d/%d predictions_n' % (len(preds_filt_n), len(preds_n)))
    cache_path_n = os.path.join('eval_results/', model_id + '_' + split + '_n.json')
    json.dump(preds_filt_n, open(cache_path_n, 'w')) # serialize to temporary json file. Sigh, COCO API...

    # Eval AllSPICE
    cocoRes_n = coco.loadRes(cache_path_n)
    cocoEvalAllSPICE = COCOEvalCapSpice(coco, cocoRes_n)
    cocoEvalAllSPICE.params['image_id'] = cocoRes_n.getImgIds()
    cocoEvalAllSPICE.evaluate()

    out = {}
    for metric, score in cocoEvalAllSPICE.eval.items():
        out['All'+metric] = score
        print('*'*10, '%s: %.3f'%('All'+metric, score), '*'*10)

    imgToEvalAllSPICE = cocoEvalAllSPICE.imgToEval
    # collect SPICE_sub_score
    for k in list(imgToEvalAllSPICE.values())[0]['SPICE'].keys():
        if k != 'All':
            out['AllSPICE_'+k] = np.array([v['SPICE'][k]['f'] for v in imgToEvalAllSPICE.values()])
            out['AllSPICE_'+k] = (out['AllSPICE_'+k][out['AllSPICE_'+k]==out['AllSPICE_'+k]]).mean()
    for p in preds_filt_n:
        image_id, caption = p['image_id'], p['caption']
        imgToEvalAllSPICE[image_id]['caption'] = capsById[image_id]
    return {'overall': out, 'imgToEvalAllSPICE': imgToEvalAllSPICE}

def eval_oracle(dataset, preds_n, model_id, split):
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_n.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()

    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]
    
    sample_n = capsById[list(capsById.keys())[0]]
    for i in range(len(capsById[list(capsById.keys())[0]])):
        preds = [_[i] for _ in capsById.values()]

        json.dump(preds, open(cache_path, 'w')) # serialize to temporary json file. Sigh, COCO API...

        cocoRes = coco.loadRes(cache_path)
        cocoEval = COCOEvalCap(coco, cocoRes)
        cocoEval.params['image_id'] = cocoRes.getImgIds()
        cocoEval.evaluate()

        imgToEval = cocoEval.imgToEval
        for img_id in capsById.keys():
            tmp = imgToEval[img_id]
            for k in tmp['SPICE'].keys():
                if k != 'All':
                    tmp['SPICE_'+k] = tmp['SPICE'][k]['f']
                    if tmp['SPICE_'+k] != tmp['SPICE_'+k]: # nan
                        tmp['SPICE_'+k] = -100
            tmp['SPICE'] = tmp['SPICE']['All']['f']
            if tmp['SPICE'] != tmp['SPICE']: tmp['SPICE'] = -100
            capsById[img_id][i]['scores'] = imgToEval[img_id]

    out = {'overall': {}, 'ImgToEval': {}}
    for img_id in capsById.keys():
        out['ImgToEval'][img_id] = {}
        for metric in capsById[img_id][0]['scores'].keys():
            if metric == 'image_id': continue
            out['ImgToEval'][img_id]['oracle_'+metric] = max([_['scores'][metric] for _ in capsById[img_id]])
            out['ImgToEval'][img_id]['avg_'+metric] = sum([_['scores'][metric] for _ in capsById[img_id]]) / len(capsById[img_id])
        out['ImgToEval'][img_id]['captions'] = capsById[img_id]
    for metric in list(out['ImgToEval'].values())[0].keys():
        if metric == 'captions':
            continue
        tmp = np.array([_[metric] for _ in out['ImgToEval'].values()])
        tmp = tmp[tmp!=-100]
        out['overall'][metric] = tmp.mean()
        print('*'*10, '%s: %.3f'%(metric, out['overall'][metric]), '*'*10)
        
    return out


# 计算Div-n和m-BLEU
def eval_div_stats(dataset, preds_n, model_id, split):
    tokenizer = PTBTokenizer()

    capsById = {}
    for i, d in enumerate(preds_n):
        d['id'] = i
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]

    n_caps_perimg = len(capsById[list(capsById.keys())[0]])  # number of captions per image
    print(n_caps_perimg)
    _capsById = capsById  # save the untokenized version
    capsById = tokenizer.tokenize(capsById)

    div_1, adiv_1 = compute_div_n(capsById, 1)
    div_2, adiv_2 = compute_div_n(capsById, 2)

    globdiv_1, _ = compute_global_div_n(capsById, 1)

    print('Diversity Statistics are as follows: \n Div1: %.2f, Div2: %.2f, gDiv1: %d\n' % (div_1, div_2, globdiv_1))

    # compute mbleu
    # 计算m-BLEU指标
    scorer = Bleu(4)
    all_scrs = []
    scrperimg = np.zeros((n_caps_perimg, len(capsById)))

    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:  # 单张图像
            tempRefsById[k] = capsById[k][:i] + capsById[k][i + 1:]  # 除了第i个caption以外的句子
            candsById[k] = [capsById[k][i]]  # 第i个caption

        score, scores = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(score)
        scrperimg[i, :] = scores[1]

    all_scrs = np.array(all_scrs)

    out = {}
    out['overall'] = {'Div1': div_1, 'Div2': div_2, 'gDiv1': globdiv_1}

    for k, score in zip(range(4), all_scrs.mean(axis=0).tolist()):
        out['overall'].update({'mBLeu_%d' % (k + 1): score})

    # 计算每个图像的m-BLEU指标
    imgToEval = {}
    for i, imgid in enumerate(capsById.keys()):
        imgToEval[imgid] = {'mBleu_2': scrperimg[:, i].mean()}
        imgToEval[imgid]['individuals'] = []
        for j, d in enumerate(_capsById[imgid]):
            imgToEval[imgid]['individuals'].append(preds_n[d['id']])
            imgToEval[imgid]['individuals'][-1]['mBleu_2'] = scrperimg[j, i]
    out['ImgToEval'] = imgToEval

    print('*'*10, 'Mean mutual Bleu scores on this set is:\nmBLeu_1, mBLeu_2, mBLeu_3, mBLeu_4','*'*10)
    print(all_scrs.mean(axis=0))

    return out

# 计算Self-CIDer分数
def eval_self_cider(dataset, preds_n, model_id, split):
    cache_path = os.path.join('eval_results/', model_id + '_' + split + '_n.json')

    coco = getCOCO(dataset)
    valids = coco.getImgIds()
    
    # Get Cider_scorer
    Cider_scorer = Cider(df='corpus')

    tokenizer = PTBTokenizer()
    gts = {}
    for imgId in valids:
        gts[imgId] = coco.imgToAnns[imgId]
    gts = tokenizer.tokenize(gts)   # 真实标注分词

    for imgId in valids:
        Cider_scorer.cider_scorer += (None, gts[imgId])
    Cider_scorer.cider_scorer.compute_doc_freq()    # 计算文档频率
    Cider_scorer.cider_scorer.ref_len = np.log(float(len(Cider_scorer.cider_scorer.crefs))) # 参考文档长度

    # Prepare captions
    capsById = {}
    for d in preds_n:
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d] # 每个图像的5个caption

    capsById = tokenizer.tokenize(capsById) # 分词
    imgIds = list(capsById.keys())          # 图像id，共N张图像
    scores = Cider_scorer.my_self_cider([capsById[_] for _ in imgIds])  # 获取每张图像的分数，即 (N, 5, 5, 1)

    def get_div(eigvals):
        eigvals = np.clip(eigvals, 0, None) # 裁剪特征值，小于0的置为0
        # r = -σmax / ∑σi
        # div = -log(r)/log(N) = - log(√λmax / ∑√λi) / log(N)
        return -np.log(np.sqrt(eigvals[-1]) / (np.sqrt(eigvals).sum())) / np.log(len(eigvals))

    # np.linalg.eigvalsh()返回的是升序排列的特征值 -> 5*5矩阵的特征值
    # get_div()计算Div-n#
    sc_scores = [get_div(np.linalg.eigvalsh(_/10)) for _ in scores] # 每张图像的分数，即（N,1)
    score = np.mean(np.array(sc_scores))    # 平均分数
    print('*'*10, 'Self-CIDer score is %.2f' % (score*100), '*'*10)
    
    imgToEval = {}
    for i, image_id in enumerate(imgIds):
        imgToEval[image_id] = {'self_cider': sc_scores[i], 'self_cider_mat': scores[i].tolist()}
    return {'overall': {'self_cider': score}, 'imgToEval': imgToEval}


# add by Greg
def eval_tags_coverage(dataset, preds_n, model_id, split, eval_kwargs):

    tags_h5 = eval_kwargs.get('tags_h5', 'data/tags_to_calculate.hdf5')  # 标签的hdf5文件路径
    tags_num = eval_kwargs.get('tags_num', 100)  # 默认计算100个标签

    tokenizer = PTBTokenizer()
    # capsById字典中，capsById字典的键是图像ID，
    # 值是一个列表，其中包含了该图像对应的所有自动生成的文本
    f = h5py.File(tags_h5, "r")

    capsById = {}
    for i, d in enumerate(preds_n):
        d['id'] = i
        capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]
    # capsById: {image_id: [caption1, caption2, ...], ...}
    # caption: {image_id: image_id, caption: caption, id: id, tags: tags}

    n_caps_perimg = len(capsById[list(capsById.keys())[0]]) # 每张图像的caption数量

    _capsById = capsById    # save the untokenized version
    capsById = tokenizer.tokenize(capsById) # capysById: {image_id: [caption1, caption2, ...], ...}
    imgIds = list(capsById.keys())
    # imgIds: [image_id1, image_id2, ...]
    print('number of images: %d' % len(imgIds))

    coverages_all = []  # 用于记录每张图像的5句话的平均覆盖率
    cov_perimg = np.zeros((len(imgIds), n_caps_perimg))# (N, 5)   # 用于记录每个图片5句话的覆盖率
    with h5py.File(tags_h5, 'r') as f:
        for i, imgid in enumerate(imgIds):
            for k in range(n_caps_perimg):
                tags_tokens = set([_.decode() for _ in f[str(imgid)]['token'][:tags_num]])  # 标签的token
                _sentence = [_ for _ in capsById[imgid][k].split()]  # 句子的token
                common_words_count = [_ for _ in _sentence if _ in tags_tokens]  # 句子中的词在标签中的数量
                cov_perimg[i, k] = len(common_words_count) / len(_sentence) \
                    if len(_sentence) > 0 else 0
                # 标注中的词在标签中的比例
            coverages_all.append(cov_perimg[i].mean())  # 每张图像的5句话的平均覆盖率
    coverages_all = np.array(coverages_all)  # (N, 1)
    score = np.mean(coverages_all)  # 平均覆盖率

    print('*'*10, 'tags coverage score is %.2f' % (score*100), '*'*10)

    imgToEval = {}
    for i, image_id in enumerate(imgIds):
        imgToEval[image_id] = {'Cov_tags': coverages_all[i]}
        imgToEval[image_id]['individuals'] = []
        for j, d in enumerate(_capsById[image_id]):
            imgToEval[image_id]['individuals'].append(preds_n[d['id']])
            imgToEval[image_id]['individuals'][-1]['Cov_tags'] = cov_perimg[i, j]

    return {'overall': {'cov_tags': score}, 'imgToEval': imgToEval}



