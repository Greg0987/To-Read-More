import argparse
from pathlib import Path
import shutil
import h5py
import numpy as np
import os

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel

import sys
# sys.path.append('')
from dataset import COCOCaptions, collate_tokens
import requests

# import time
# time.sleep(0.5) # 放慢请求问题速度

class TagDB(LightningModule):   # 用于获取text特征
    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = save_dir
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # 使用clip模型
        self.clip.eval()  # 设置为验证模式
        for p in self.clip.parameters():  # 禁止梯度传播（即锁住参数）
            p.requires_grad = False

    def test_step(self, batch, batch_idx):
        """
        batch: [captions, tokens]
        """
        if batch is None:
            return None

        captions, tokens = batch  # 从batch中获取描述和标记
        tokens = {k: v.to(self.device) for k, v in tokens.items()}  # 建立tokens的字典{k:v}

        # 此处features是clip模型输出的pooler_output，代表整个序列的向量表示，只是最后一层的隐藏状态
        features = self.clip.text_model(**tokens)[1]  # 用CLIP的文本模型，获取tokens的文本特征
        # 此处keys是features经由projection layer和归一化。是NLP任务中的最终结果表示向量，可以视为是text_embeds
        keys = self.clip.text_projection(features)
        keys = keys / keys.norm(dim=-1, keepdim=True)  # 归一化操作

        # keys用于计算text和image之间的相似度
        # features用于计算text之间的相似度
        features = features.detach().cpu().numpy()
        keys = keys.detach().cpu().numpy()

        with h5py.File(os.path.join(self.save_dir, f"tags_db_{args.id}.hdf5"), "a") as f:  # 利用h5py库对数据集/权重、偏差等参数保存为.h5文件
            # outputs/tags_db/tags_db.hdf5
            g = f.create_group(str(batch_idx))
            g.create_dataset("keys", data=keys, compression="gzip")
            g.create_dataset("features", data=features, compression="gzip")
            g.create_dataset("tags", data=captions, compression="gzip")

def build_tag_db(args):
    dset = COCOCaptions(os.path.join(args.ann_dir, "annotations"), args.vocab_dir)   # COCO描述数据类
    dloader = DataLoader(  # 数据加载器
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_tokens,
        persistent_workers=True
    )
    cap_db = TagDB(args.save_dir)  # 建立描述类

    trainer = Trainer(  # 训练器
        # gpus=[args.device, ],
        accelerator='gpu',
        devices=args.device,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(cap_db, dloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode tags')  # 参数设置
    # parser.add_argument('--device', type=int, default=[0])
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--id', type=str, default='9486')
    parser.add_argument('--exp_name', type=str, default='tags_db')
    parser.add_argument('--ann_dir', type=str, default='data/coco_captions')
    parser.add_argument('--vocab_dir', type=str, default='data/cocotalk.json')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=7)
    args = parser.parse_args()

    # args.dataset_root = Path(args.dataset_root)  # 数据集根路径
    setattr(args, "save_dir", Path("data") / args.exp_name)  # 设置属性save_dir值
    shutil.rmtree(args.save_dir, ignore_errors=True)  # 递归删除文件夹下的所有子文件夹和子文件
    args.save_dir.mkdir(parents=True, exist_ok=True)  # 创建该目录
    print(args)  # 打印参数

    seed_everything(1, workers=True)  # 随机种子

    build_tag_db(args)  # 建立描述
