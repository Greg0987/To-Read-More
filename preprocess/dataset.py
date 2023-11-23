"""
[1]: 当前事项
[2]: 表示下述代码已修改，无问题
[2]~[3]区间表示代码已检查
[3]: 表示上述代码已修改，无问题
[4]~[8]存在问题待改

"""
import sys
import os

sys.path.append(os.getcwd())
# sys.path.append('/home/ps/DiT')

from PIL import Image
from pathlib import Path
import numpy as np
from coco_caption.pycocotools.coco import COCO as pyCOCO
import json
import string
from tqdm import tqdm
import itertools

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor
from torchvision.transforms import functional as F


from cider.pyciderevalcap.tokenizer.ptbtokenizer import PTBTokenizer

LENGTH_LIMIT = 75

# 定义将多个样本组合成一个批次（预处理，使其符合神经网络的输入格式）
def collate_tokens(batch):
    captions, input_ids, attention_mask, lengths = [], [], [], []
    for cap, tok in batch:
        assert tok["input_ids"].shape == tok["attention_mask"].shape
        captions.append(cap)

        # 对token进行长度限制
        l = tok["input_ids"].shape[1]
        if l < LENGTH_LIMIT:
            input_ids.append(tok["input_ids"])
            attention_mask.append(tok["attention_mask"])
            lengths.append(l)
        else:
            input_ids.append(tok["input_ids"][:,:LENGTH_LIMIT])
            attention_mask.append(tok["attention_mask"][:,:LENGTH_LIMIT])
            lengths.append(LENGTH_LIMIT)

    # padding到最大长度
    max_len = max(lengths)
    input_pad, atten_pad = [], []
    for i in range(len(input_ids)):
        l = input_ids[i].shape[1]
        if l < max_len:
            p = torch.zeros(size=(1, max_len-l), dtype=input_ids[i].dtype)  # padding
            input_pad.append(torch.cat([input_ids[i],p], dim=1))    # 对input进行填充

            p = torch.zeros(size=(1, max_len-l), dtype=attention_mask[i].dtype)
            atten_pad.append(torch.cat([attention_mask[i], p], dim=1))
        else:
            input_pad.append(input_ids[i])
            atten_pad.append(attention_mask[i])

    # 将v进行concat
    input_pad = torch.cat(input_pad)
    atten_pad = torch.cat(atten_pad)
    assert input_pad.shape[1] <= LENGTH_LIMIT
    assert atten_pad.shape[1] <= LENGTH_LIMIT
    assert input_pad.shape == atten_pad.shape

    tokens = {"input_ids": input_pad, "attention_mask":atten_pad}

    return captions, tokens

# COCO描述数据集类
class COCOCaptions(Dataset):
    def __init__(self, ann_dir, vocab_dir):
        super().__init__()
        # CLIP分词器
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
        # 用于替换ASCII控制字符0-31
        escapes = ''.join([chr(char) for char in range(0, 32)])
        self.escapes = escapes
        self.translator = str.maketrans('', '', escapes)
        punctuation_list = string.punctuation
        self.punctuation = punctuation_list
        self.vocab_dir = vocab_dir

        self.tags = self.parse_annotations(ann_dir)   # 从路径中解析出注释（描述）

    @staticmethod
    def combination(l1, l2):
        return [" ".join(x) for x in itertools.product(l1, l2)]
    # 去除控制字符
    def process_word(self, s):
        return s.lower().strip().translate(self.translator)
    # 去除标点符号
    def process_punctuation(self, s):
        return s.translate(self.punctuation)

    # 处理同义词，只取第一个
    def process_synset(self, s):
        return s.lower().strip().translate(self.translator).split(".")[0]

    # 解析出注释
    def parse_annotations(self, ann_dir):
        """
        【此处有问题待改！！！！！！
        不能简单把word分成一个一个字，要用分词？？
        待思考英文要不要拆分分词
        对于单个单词，输入CLIP模型中可能效果不好】
        """
        tokenizer = PTBTokenizer()
        tags = set()

        print("parsing tags")
        # print("parsing train captions")
        # with open(os.path.join(ann_dir, "captions_train2014.json"), "r") as f:
        #     captions_train = json.load(f)
        # # for x in captions_train['annotations']:
        # #     caption = x['caption'].strip().translate(str.maketrans('', '', self.punctuation))
        # #     caption = caption.translate(str.maketrans('','', self.escapes))
        # #
        # #     for word in caption.split():
        # #         tags.add(word)
        # #     # print(tags)
        # capsById = {}
        # for i, d in enumerate(captions_train['annotations']):
        #     d['id'] = i
        #     capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]
        # # capsById: {image_id: [caption1, caption2, ...]}
        # # caption: {'image_id': 318556, 'id': 0, 'caption': 'A very clean and well decorated empty bathroom'}
        # capsById = tokenizer.tokenize(capsById)
        # for k, v in capsById.items():
        #     for tokenized_caption in v:
        #         for word in tokenized_caption.split():
        #             tags.add(word)
        #
        # print("parsing val captions")
        # with open(os.path.join(ann_dir,"captions_val2014.json"), "r") as f:
        #     captions_val = json.load(f)
        # # for x in captions_val['annotations']:
        # #     caption = x['caption'].strip().translate(str.maketrans('', '', self.punctuation))
        # #     caption = caption.translate(str.maketrans('', '', self.escapes))
        # #     for word in caption.split():
        # #         tags.add(word)
        # capsById = {}
        # for i, d in enumerate(captions_val['annotations']):
        #     d['id'] = i
        #     capsById[d['image_id']] = capsById.get(d['image_id'], []) + [d]
        # capsById = tokenizer.tokenize(capsById)
        # for k, v in capsById.items():
        #     for tokenized_caption in v:
        #         for word in tokenized_caption.split():
        #             tags.add(word)

        # print(len(tags), tags)

        vocab = json.load(open(self.vocab_dir, "r"))
        vocab = vocab['word_to_ix']
        del vocab['UNK']
        _tags = set()
        for word, ix in vocab.items():
            _tags.add(word)

        tags = _tags
        tags = np.unique(list(tags)).tolist()
        print("The length of tags is: ", len(tags))
        return tags

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, index):
        tokens = self.tokenizer(self.tags[index], padding=True, return_tensors="pt")
        return self.tags[index], tokens

# 整理图像分块，按块别，9块：[图1，图2，...]；16块：[图1，图2，...]；25块：[图1，图2，...]
def collate_crops(data):
    orig_image, four_images, nine_images, sixteen_images, twentyfive_images, captions, idx = zip(*data) # [3,224,224]

    orig_image = torch.stack(list(orig_image), dim=0)   # [b_s, 3, 224, 224]
    four_images = torch.stack(list(four_images), dim=0) # [b_s, 4, 3, 224, 224]
    nine_images = torch.stack(list(nine_images), dim=0) # [b_s, 9, 3, 224, 224]
    sixteen_images = torch.stack(list(sixteen_images), dim=0)
    twentyfive_images = torch.stack(list(twentyfive_images), dim=0)
    captions = list(captions)
    idx = torch.LongTensor(list(idx))

    return orig_image, four_images, nine_images, sixteen_images, twentyfive_images, captions, idx

# COCO图像分块数据集
class CocoImageCrops(Dataset):
    def __init__(self, ann_dir, img_root, transform=None):
        self.transform = transform  # 图像预处理函数，从encode_images.py导入
        self.data = self.parse(Path(ann_dir), Path(img_root))

    # 对数据进行解析，打包返回data
    """
    data[img_id]{"image_id""image_file""captions"}
    """
    @staticmethod
    def parse(ann_dir, img_root):
        ids = (
            np.load(ann_dir / "coco_train_ids.npy"),
            np.concatenate([
                np.load(ann_dir / "coco_restval_ids.npy"),
                np.load(ann_dir / "coco_dev_ids.npy"),
                np.load(ann_dir / "coco_test_ids.npy")
            ]),
        )
        coco = (
            pyCOCO(ann_dir / "captions_train2014.json"),
            pyCOCO(ann_dir / "captions_val2014.json"),
        )
        img_root = (img_root / "train2014", img_root / "val2014")

        # 将id, image, capttions打包
        data = {}
        for i in range(len(ids)):
            for idx in ids[i]:
                img_id = coco[i].anns[idx]["image_id"]
                img_file = img_root[i] / coco[i].loadImgs(img_id)[0]["file_name"]
                caption = coco[i].anns[idx]["caption"].strip()

                if img_id in data:
                    data[img_id]["captions"].append(caption)
                else:
                    data[img_id] = {
                        "image_id": img_id,
                        "image_file": img_file,
                        "captions": [caption, ]
                    }

        data = list(data.values())
        data.sort(key=lambda x: x["image_id"])

        return data

    def four_crop(self, image):
        w, h = image.size
        ww, hh = w/2, h/2
        w = [(0, ww), (ww, 2*ww)]
        h = [(0, hh), (hh, 2*hh)]
        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            images.append(F.crop(image, top, left, hh, ww))
        return images

    def nine_crop(self, image):
        w, h = image.size
        ww, hh = w/3, h/3
        w = [(0, ww), (ww, 2*ww), (2*ww, 3*ww)]
        h = [(0, hh), (hh, 2*hh), (2*hh, 3*hh)]
        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            images.append(F.crop(image, top, left, hh, ww))
        return images

    def sixteen_crop(self, image):
        w, h = image.size
        ww, hh = w/4, h/4
        w = [(0, ww), (ww, 2*ww), (2*ww, 3*ww), (3*ww, 4*ww)]
        h = [(0, hh), (hh, 2*hh), (2*hh, 3*hh), (3*hh, 4*hh)]
        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            images.append(F.crop(image, top, left, hh, ww))
        return images

    def twentyfive_crop(self, image):
        w, h = image.size
        ww, hh = w / 5, h / 5
        w = [(0, ww), (ww, 2 * ww), (2 * ww, 3 * ww), (3 * ww, 4 * ww), (4*ww, 5*ww)]
        h = [(0, hh), (hh, 2 * hh), (2 * hh, 3 * hh), (3 * hh, 4 * hh), (4*hh, 5*hh)]
        images = []
        for s in itertools.product(h, w):
            h, w = s
            top, left = h[0], w[0]
            images.append(F.crop(image, top, left, hh, ww))
        return images

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(self.data[index]["image_file"])
        image = image.convert("RGB")

        four_images = self.four_crop(image)
        nine_images = self.nine_crop(image)
        sixteen_images = self.sixteen_crop(image)
        twentyfive_images = self.twentyfive_crop(image)

        # 进行图像预处理
        if self.transform is not None:
            orig_image = self.transform(image)  # [3, 224, 224]
            four_images = torch.stack([self.transform(x) for x in four_images]) # [4, 3, 224, 224]
            nine_images = torch.stack([self.transform(x) for x in nine_images]) # [9, 3, 224, 224]
            sixteen_images = torch.stack([self.transform(x) for x in sixteen_images])
            twentyfive_images = torch.stack([self.transform(x) for x in twentyfive_images])

        captions = self.data[index]["captions"]
        idx = self.data[index]["image_id"]

        return orig_image, four_images, nine_images, sixteen_images, twentyfive_images, captions, idx


