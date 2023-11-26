import argparse
from pathlib import Path
import h5py
import numpy as np
import math
import faiss    # 可以设置索引的数据库
from tqdm import tqdm
import shutil

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from pytorch_lightning import Trainer, LightningModule, seed_everything
from transformers import CLIPModel, CLIPProcessor
import json

import sys
sys.path.append('..')
from dataset import CocoImageCrops, collate_crops

max_batches = 5

# 检索tags
class TagsRetriever(LightningModule):
    def __init__(self, tag_db, save_dir, k, num, input_json):
        super(TagsRetriever, self).__init__()

        self.save_dir = Path(save_dir)
        self.k = k
        self.num = num
        self.vocab = json.load(open(input_json, "r"))["word_to_ix"]

        self.keys, self.features, self.text = self.load_tag_db(tag_db)
        self.index = self.build_index(idx_file = self.save_dir / "faiss.index")
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    @staticmethod
    def load_tag_db(tag_db):
        print("Loading tag db")
        keys, features, text = [], [], []
        with h5py.File(tag_db, "r") as f:
            for i in tqdm(range(len(f))):
                keys_i = f[f"{i}/keys"][:]
                features_i = f[f"{i}/features"][:]
                text_i = [str(x, "utf-8") for x in f[f"{i}/tags"][:]]

                keys.append(keys_i)
                features.append(features_i)
                text.extend(text_i)
        keys = np.concatenate(keys)
        features = np.concatenate(features)
        return keys, features, text # 获取对应keys和features

    # 建立索引，并返回其index
    def build_index(self, idx_file):
        print("Building db index")
        n, d = self.keys.shape
        K = round(8 * math.sqrt(n)) # 返回四舍五入值，8*√n，聚类中心数量
        # 创建索引
        # 先进行聚类的倒排索引，支持欧式距离和向量内积两种距离算法
        index = faiss.index_factory(d, f"IVF{K},Flat",
                                    faiss.METRIC_INNER_PRODUCT)

        assert not index.is_trained # 确保索引未被训练
        index.train(self.keys)  # 训练索引
        assert index.is_trained # 确保索引已训练
        index.add(self.keys)    # 再添加向量
        index.nprobe = max(1, K//10)    # 选择最大K//10个空间进行索引

        faiss.write_index(index, str(idx_file))  # 导出索引

        return index

    def search(self, images, topk):
        features = self.clip.vision_model(pixel_values=images)[1] # 将imagesclip视觉模型，获取输出的第一维度为特征
        query = self.clip.visual_projection(features)   # 将特征输入视觉推断，获取q
        query = query / query.norm(dim=-1, keepdim=True)  # q单位化
        D, I = self.index.search(query.detach().cpu().numpy(), topk)  # 对q进行查询，获取topk个索引

        return D, I
        # D表示搜索结果与原向量的距离数据组，即近邻向量到原向量的距离由小到大的排列
        # I表示搜索结果与原向量的index

    def test_step_v2(self, batch, batch_idx):    # 已写待检查
        # if batch_idx >= max_batches:
        #     self.trainer.should_stop = True
        #     return

        orig_image, nine_images, sixteen_images, twentyfive_images, captions, idx = batch
        N = len(orig_image) # 共有N张图片

        with h5py.File(self.save_dir / f"img_tags_{self.k}.hdf5", "a")as f:
            D_o, I_o = self.search(orig_image, topk=self.k)  # N * k 个标签

            D_n, I_n = self.search(torch.flatten(nine_images, end_dim=1), topk=self.k)
            D_n, I_n = D_n.reshape(N, 9, self.k), I_n.reshape(N, 9, self.k) # N * 9 * k个标签

            D_st, I_st = self.search(torch.flatten(sixteen_images, end_dim=1), topk=self.k)
            D_st, I_st = D_st.reshape(N, 16, self.k), I_st.reshape(N, 16, self.k)  # N * 9 * k个标签

            D_tf, I_tf = self.search(torch.flatten(twentyfive_images, end_dim=1), topk=self.k)
            D_tf, I_tf = D_tf.reshape(N, 25, self.k), I_tf.reshape(N, 25, self.k)  # N * 9 * k个标签

            for i in range(N):
                dict_tags = {}


                g1 = f.create_group(str(int(idx[i])))   # 图片目录：img_id

                tags = [self.text[j] for j in I_o[i]]   # 单张图片对应的tags
                features = self.features[I_o[i]]        # 单张图片，对应的k个tags的features，[k, d]
                scores = D_o[i]                         # 分数，查询向量的距离
                values = list(zip(scores, features))
                for j in range(len(tags)):
                    if tags[j] in dict_tags:
                        dict_tags[tags[j]] = (max(dict_tags[tags[j]][0], values[j][0]), \
                                              values[j][1])
                    else:
                        dict_tags[tags[j]] = values[j]

                # g2 = g1.create_group("whole")                   # 打包成组，单块crop的目录
                # g2.create_dataset("features", data=features)
                # g2.create_dataset("scores", data=scores)
                # g2.create_dataset("tags", data=tags)

                tags = [[self.text[I_n[i, j, k]] for k in range(self.k)] for j in range(9)] # i * j * k = N * 9 * k
                features = self.features[I_n[i].flatten()].reshape((9, self.k, -1))
                scores = D_n[i]
                for c in range(9):
                    values = list(zip(scores[c], features[c]))
                    for j in range(len(tags[0])):
                        if tags[c][j] in dict_tags:
                            dict_tags[tags[c][j]] = (max(dict_tags[tags[c][j]][0], values[j][0]), \
                                                     values[j][1])
                        else:
                            dict_tags[tags[c][j]] = values[j]


                # g3 = g1.create_group("nine")
                # g3.create_dataset("features", data=features)
                # g3.create_dataset("scores", data=scores)
                # g3.create_dataset("tags", data=tags)

                tags = [[self.text[I_st[i, j, k]] for k in range(self.k)] for j in range(16)]  # i * j * k = N * 16 * k
                features = self.features[I_st[i].flatten()].reshape((16, self.k, -1))
                scores = D_st[i]
                for c in range(16):
                    values = list(zip(scores[c], features[c]))
                    for j in range(len(tags[0])):
                        if tags[c][j] in dict_tags:
                            dict_tags[tags[c][j]] = (max(dict_tags[tags[c][j]][0], values[j][0]), \
                                                     values[j][1])
                        else:
                            dict_tags[tags[c][j]] = values[j]
                # g4 = g1.create_group("sixteen")
                # g4.create_dataset("features", data=features)
                # g4.create_dataset("scores", data=scores)
                # g4.create_dataset("tags", data=tags)

                tags = [[self.text[I_tf[i, j, k]] for k in range(self.k)] for j in range(25)]  # i * j * k = N * 25 * k
                features = self.features[I_tf[i].flatten()].reshape((25, self.k, -1))
                scores = D_tf[i]
                for c in range(25):
                    values = list(zip(scores[c], features[c]))
                    for j in range(len(tags[0])):
                        if tags[c][j] in dict_tags:
                            dict_tags[tags[c][j]] = (max(dict_tags[tags[c][j]][0], values[j][0]), \
                                                     values[j][1])
                        else:
                            dict_tags[tags[c][j]] = values[j]


                sorted_list_tuple = sorted(dict_tags.items(), key=lambda x: x[1][0], reverse=True)
                sorted_list = []
                for (x, (y, z)) in sorted_list_tuple:
                    sorted_list.append([x, y, z])

                cut_list = sorted_list[:100]
                # [tags, scores, features]
                tags = [row[0] for row in cut_list]
                scores = [row[1] for row in cut_list]
                features = [row[2] for row in cut_list]
                g1.create_dataset("tags", data=tags)
                g1.create_dataset("scores", data=scores)
                g1.create_dataset("features", data=features)

                # g5 = g1.create_group("twentyfive")
                # g5.create_dataset("features", data=features)
                # g5.create_dataset("scores", data=scores)
                # g5.create_dataset("tags", data=tags)

    def test_step(self, batch, batch_idx):    # 已写待检查
        # if batch_idx >= max_batches:
        #     self.trainer.should_stop = True
        #     return

        orig_image, four_images, nine_images, sixteen_images, twentyfive_images, captions, idx = batch
        N = len(orig_image) # 共有N张图片

        with h5py.File(self.save_dir / f"img_tags_top{self.k}.hdf5", "a") as f:
            with h5py.File(self.save_dir / "tags_to_calculate.hdf5", "a") as ff:
                D_o, I_o = self.search(orig_image, topk=self.k)  # N * k 个标签

                D_f, I_f = self.search(torch.flatten(four_images, end_dim=1), topk=self.k)
                D_f, I_f = D_f.reshape(N, 4, self.k), I_f.reshape(N, 4, self.k) # N * 4 * k个标签

                D_n, I_n = self.search(torch.flatten(nine_images, end_dim=1), topk=self.k)
                D_n, I_n = D_n.reshape(N, 9, self.k), I_n.reshape(N, 9, self.k) # N * 9 * k个标签

                D_st, I_st = self.search(torch.flatten(sixteen_images, end_dim=1), topk=self.k)
                D_st, I_st = D_st.reshape(N, 16, self.k), I_st.reshape(N, 16, self.k)  # N * 9 * k个标签

                D_tf, I_tf = self.search(torch.flatten(twentyfive_images, end_dim=1), topk=self.k)
                D_tf, I_tf = D_tf.reshape(N, 25, self.k), I_tf.reshape(N, 25, self.k)  # N * 9 * k个标签

                for i in range(N):
                    dict_tags = {}
                    g1 = f.create_group(str(int(idx[i])))   # 图片目录：img_id

                    tags = [self.text[j] for j in I_o[i]]   # 单张图片对应的tags
                    features = self.features[I_o[i]]        # 完整图片，对应的k个tags的features，[k, d]
                    scores = D_o[i]                         # 分数，查询向量的距离
                    for j in range(len(tags)):
                        if tags[j] in dict_tags:
                            dict_tags[tags[j]] = max(dict_tags[tags[j]], scores[j])
                        else:
                            dict_tags[tags[j]] = scores[j]
                    g2 = g1.create_group("whole")                   # 打包成组，单块crop的目录
                    g2.create_dataset("features", data=features)
                    g2.create_dataset("scores", data=scores)
                    g2.create_dataset("tags", data=tags)

                    tags = [[self.text[I_f[i, j, k]] for k in range(self.k)] for j in range(4)] # i * j * k = N * 4 * k
                    features = self.features[I_f[i].flatten()].reshape((4, self.k, -1))
                    scores = D_f[i]
                    for c in range(4):
                        for j in range(len(tags[0])):
                            if tags[c][j] in dict_tags:
                                dict_tags[tags[c][j]] = max(dict_tags[tags[c][j]], scores[c][j])
                            else:
                                dict_tags[tags[c][j]] = scores[c][j]
                    g3 = g1.create_group("four")
                    g3.create_dataset("features", data=features)
                    g3.create_dataset("scores", data=scores)
                    g3.create_dataset("tags", data=tags)


                    tags = [[self.text[I_n[i, j, k]] for k in range(self.k)] for j in range(9)] # i * j * k = N * 9 * k
                    features = self.features[I_n[i].flatten()].reshape((9, self.k, -1))
                    scores = D_n[i]
                    for c in range(9):
                        for j in range(len(tags[0])):
                            if tags[c][j] in dict_tags:
                                dict_tags[tags[c][j]] = max(dict_tags[tags[c][j]], scores[c][j])
                            else:
                                dict_tags[tags[c][j]] = scores[c][j]
                    g4 = g1.create_group("nine")
                    g4.create_dataset("features", data=features)
                    g4.create_dataset("scores", data=scores)
                    g4.create_dataset("tags", data=tags)

                    tags = [[self.text[I_st[i, j, k]] for k in range(self.k)] for j in range(16)]  # i * j * k = N * 16 * k
                    features = self.features[I_st[i].flatten()].reshape((16, self.k, -1))
                    scores = D_st[i]
                    for c in range(16):
                        for j in range(len(tags[0])):
                            if tags[c][j] in dict_tags:
                                dict_tags[tags[c][j]] = max(dict_tags[tags[c][j]], scores[c][j])
                            else:
                                dict_tags[tags[c][j]] = scores[c][j]
                    g5 = g1.create_group("sixteen")
                    g5.create_dataset("features", data=features)
                    g5.create_dataset("scores", data=scores)
                    g5.create_dataset("tags", data=tags)

                    tags = [[self.text[I_tf[i, j, k]] for k in range(self.k)] for j in range(25)]  # i * j * k = N * 25 * k
                    features = self.features[I_tf[i].flatten()].reshape((25, self.k, -1))
                    scores = D_tf[i]
                    for c in range(25):
                        for j in range(len(tags[0])):
                            if tags[c][j] in dict_tags:
                                dict_tags[tags[c][j]] = max(dict_tags[tags[c][j]], scores[c][j])
                            else:
                                dict_tags[tags[c][j]] = scores[c][j]
                    g6 = g1.create_group("twentyfive")
                    g6.create_dataset("features", data=features)
                    g6.create_dataset("scores", data=scores)
                    g6.create_dataset("tags", data=tags)

                    sorted_list_tuple = sorted(dict_tags.items(), key=lambda x: x[1], reverse=True) # 按照分数排序
                    sorted_list = []
                    for (tag, score) in sorted_list_tuple:
                        sorted_list.append(tag)         # 按照分数排序的tags
                    cut_list = sorted_list[:self.num]   # 取前num个tags
                    tags = [x for x in cut_list]        # 选取的tags
                    ids = [self.vocab.get(x) for x in tags if x in self.vocab]   # 0为<unk>

                    ff1 = ff.create_group(str(int(idx[i])))
                    ff1.create_dataset("token", data=tags)
                    ff1.create_dataset("id", data=ids)


def build_tags(args):
    transform = T.Compose([
        # 特征提取
        CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").feature_extractor,
        # 获取pixel_values第0维为长张量
        lambda x: torch.FloatTensor(x["pixel_values"][0]),
    ])
    dset = CocoImageCrops(args.dataset_root/"annotations", args.dataset_root, transform)
    dloader = DataLoader(
        dataset=dset,
        batch_size =args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        collate_fn=collate_crops,
        persistent_workers=True
    )

    tag_retr = TagsRetriever(
        tag_db=args.tag_db,
        save_dir=args.save_dir,
        k=args.k,
        num=args.num,
        input_json=args.input_json,
    )

    trainer = Trainer(
        accelerator='gpu',
        devices=args.device,
        deterministic=True,
        benchmark=False,
        default_root_dir=args.save_dir,
        limit_test_batches=args.limit_batches
    )
    trainer.test(tag_retr, dloader) # 对COCO数据集的annotations进行检索


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve tags')
    # parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--device', type=str, default="0")
    parser.add_argument('--exp_name', type=str, default='retrieve_tags')
    parser.add_argument('--dataset_root', type=str, default='data/coco_captions')
    parser.add_argument('--input_json', default='data/cocotalk.json', help='input json file of vocalbulary')
    parser.add_argument('--tag_db', type=str, default='data/tags_db/tags_db_9486.hdf5')
    parser.add_argument('--k', type=int, default=20, help="top k tags")
    parser.add_argument('--num', type=int, default=100, help="num of tags to calculate")
    parser.add_argument('--limit_batches', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=12)
    args = parser.parse_args()

    args.dataset_root = Path(args.dataset_root)
    setattr(args, "save_dir", Path("data") / args.exp_name)
    shutil.rmtree(args.save_dir, ignore_errors=True)
    args.save_dir.mkdir(parents=True, exist_ok=True)
    print(args)

    seed_everything(1, workers=True)

    build_tags(args)
