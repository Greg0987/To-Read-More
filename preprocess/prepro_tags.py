
import os
import json
import argparse
from random import shuffle, seed
import h5py
import numpy as np
# 导入进度条
from tqdm import tqdm

def main(params):
    vocab = json.load(open(params['input_json'], 'r'))['word_to_ix']
    tags_file = params['input_hdf5']
    output_file = params['output_dir']
    num = params['num']

    crops = ['whole', 'four', 'nine', 'sixteen', 'twentyfive']

    if h5py.is_hdf5(output_file):
        os.remove(output_file)

    with h5py.File(tags_file, 'r') as f1:
        with h5py.File(output_file, 'a') as f2:
            for id in tqdm(f1):
                dict_tags = {}  # 每张图片的tags
                for c in crops:
                    tags = f1[id][c]["tags"]
                    scores = f1[id][c]["scores"]
                    if c == 'whole':
                        for i in range(len(tags)):
                            if tags[i] in dict_tags:
                                dict_tags[tags[i]] = max(dict_tags[tags[i]], scores[i])
                            else:
                                dict_tags[tags[i]] = scores[i]
                    else:
                        for i in range(len(tags)):
                            for j in range(len(tags[i])):
                                if tags[i][j] in dict_tags:
                                    dict_tags[tags[i][j]] = max(dict_tags[tags[i][j]], scores[i][j])
                                else:
                                    dict_tags[tags[i][j]] = scores[i][j]
                sorted_list_tuple = sorted(dict_tags.items(), key=lambda x: x[1], reverse=True)
                sorted_list = []
                for (k, v) in sorted_list_tuple:
                    sorted_list.append([k, v])
                cut_list = sorted_list[:num]
                tags = [x[0] for x in cut_list]
                ids = [vocab[x.decode()] for x in tags]

                g1 = f2.create_group(id) # 创建组：图像id
                g1.create_dataset('token', data=tags)
                g1.create_dataset('id', data=ids)

                print(id, cut_list[:5])
    print('wrote', output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input json
    parser.add_argument('--input_json', default='data/cocotalk.json', help='input json file of vocalbulary')
    parser.add_argument('--input_hdf5', default='data/img_tags_top20.hdf5', help='input tags file to calc')
    parser.add_argument('--output_dir', default='data/tags_to_calculate.hdf5', help='output h5 file')

    parser.add_argument('--num', default=100, type=int, help='number of tags left to calculate')

    args = parser.parse_args()
    params = vars(args) # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)