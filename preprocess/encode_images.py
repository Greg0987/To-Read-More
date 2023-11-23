
import argparse # 参数库
from pathlib import Path    # 路径库
import h5py # 存储库
import shutil   # 文件操作库


import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T # 对图片进行变换
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor   # CLIP模型

import sys
sys.path.append('..')
from dataset import CocoImageCrops, collate_crops


# 图像编码类
class ImageEncoder(LightningModule):  # 图像编码器
    def __init__(self, save_dir):
        super().__init__()

        self.save_dir = Path(save_dir)  # 保存路径
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model  # 使用CLIP的视觉模型

    def test_step(self, batch, batch_idx):
        """batch: [imgs, *, *, *, id]  此处是一个batch的序列，需要细分"""
        orig_image, four_images, nine_images, sixteen_images, tweentyfive_images, captions, idx = batch  # 获取原始图像和对应id

        b_s, c, h, w = orig_image.size()
        # [b_s, 3, 224, 224] -> [b_s, 768]
        features_orig = self.model(pixel_values=orig_image)
        """
        将图像输入CLIP视觉模型获取其特征
        此处指输出序列的第一个token的最后一个隐藏层状态，即CLS标签的embedding，又经历一次全连接层输出
        可以用pooler_output结果作baseline
        可以理解为该句子语义的特征向量表示
        """
        features_orig = features_orig.pooler_output # [bs, hidden_size]
        output_shape = features_orig.size() # [b_s, 768]
        features_orig = torch.unsqueeze(features_orig, 1) # -> [b_s, 1, 768]
        features_orig = features_orig.detach().cpu().numpy()

        four_images = four_images.reshape(-1, c, h, w)  # -> [b_s * 9, c, h, w]
        features_four = self.model(pixel_values=four_images).pooler_output   # -> [b_s * 9, 768]
        features_four = features_four.reshape(b_s, -1, output_shape[-1]).detach().cpu().numpy()

        nine_images = nine_images.reshape(-1, c, h, w)  # -> [b_s * 9, c, h, w]
        features_nine = self.model(pixel_values=nine_images).pooler_output   # -> [b_s * 9, 768]
        features_nine = features_nine.reshape(b_s, -1, output_shape[-1]).detach().cpu().numpy()

        sixteen_images = sixteen_images.reshape(-1, c, h, w)  # -> [b_s * 9, c, h, w]
        features_sixteen = self.model(pixel_values=sixteen_images).pooler_output   # -> [b_s * 9, 768]
        features_sixteen = features_sixteen.reshape(b_s, -1, output_shape[-1]).detach().cpu().numpy()

        tweentyfive_images = tweentyfive_images.reshape(-1, c, h, w)  # -> [b_s * 9, c, h, w]
        features_tweentyfive = self.model(pixel_values=tweentyfive_images).pooler_output  # -> [b_s * 9, 768]
        features_tweentyfive = features_tweentyfive.reshape(b_s, -1, output_shape[-1]).detach().cpu().numpy()

        # with h5py.File(self.save_dir / "img_grids.hdf5", "a") as f:
        # f = h5py.File(self.save_dir / "img_grids.hdf5", "a")


        # f.attrs["fdim"] = features_orig.shape[-1]  # 特征的维度

        with h5py.File(self.save_dir / "img_grids.hdf5", "a") as f:
            f.attrs["fdim"] = features_orig.shape[-1]
            for i in range(len(orig_image)):
                # f.create_dataset(str(int(ids[i])), data=features[i])  # 有多少张照片就对应特征
                g1 = f.create_group(str(int(idx[i])))

                g1.create_dataset("whole", data=features_orig[i])   # [1, 768]
                g1.create_dataset("four", data=features_four[i])    # [4, 768]
                g1.create_dataset("nine", data=features_nine[i])    # [9, 768]
                g1.create_dataset("sixteen", data=features_sixteen[i])  # [16, 768]
                g1.create_dataset("tweentyfive", data=features_tweentyfive[i])  # [25, 768]


def extract_image_features(x):
    return torch.FloatTensor(x["pixel_values"][0])

# 图像编码
def build_image_features(args):
    # 组合操作：对图像进行预处理（单个图像）
    transform = T.Compose([
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,    # 特征提取
        extract_image_features,  # 建立张量
    ])

    dset = CocoImageCrops(args.dataset_root/"annotations", args.dataset_root, transform)    # 建立COCO数据集
    dloader = DataLoader(   # 数据加载器
        dataset=dset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops,
        persistent_workers=True
    )

    img_encoder = ImageEncoder(args.save_dir)  # 建立图像编码模型类

    trainer = Trainer(  # 训练器
        # gpus = None,
        accelerator='gpu',
        devices=args.device,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir
    )
    trainer.test(img_encoder, dloader)  # 对图像用encoder进行编码


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Encode images')  # 设置参数
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--exp_name', type=str, default='image_features')
    parser.add_argument('--dataset_root', type=str, default='data/coco_captions')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)  # 数据集根路径
    setattr(args, "save_dir", Path("data") / args.exp_name)  # 设置属性save_dir的值
    shutil.rmtree(args.save_dir, ignore_errors=True)  # 递归删除文件夹下的所有子文件夹和子文件
    args.save_dir.mkdir(parents=True, exist_ok=True)  # 创建该目录
    print(args)

    seed_everything(1, workers=True)  # 随机数种子

    build_image_features(args)  # 进行图像编码操作