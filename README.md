# To-Read-More
The Code and Data of the paper To Read More.

# 1. Preprocess data of MS-COCO
Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

Download the coco images from [link](http://mscoco.org/dataset/#download). We need 2014 training images and 2014 val. images. You should put the `train2014/` and `val2014/` in the same directory in `data/`.

## 1.1 Encode images
```bash
$ python preprocess/encode_images.py
```

## 1.2 Encode tags data
```bash
$ python preprocess/encode_captions.py
```

## 1.3 Consturct corresponding pseudo tags
```bash
$ python preprocess/retrieve_tags.py
```

## 1.4 Download the object features
Download from [link](https://www.dropbox.com/s/0h67c6ezwnderbd/oscar.hdf5) which is training on weaker Visual Genome offered from Xmodal-CTX.

And then make sure all the files generated before are in `data/`.
`cocotalk.json`
`cocotalk_label.h5`
`dataset_coco.json`
`img_grids.hdf5`
`img_tags_top20.hdf5`
`oscar.hdf5`
`tags_to_calculate.hdf5`

# 2 Start training
## 2.1 XE
Set your model name like 'M9'.
```bash
$ python tools/train.py --cfg configs/DiM2T.yml --id M9
```
Then you can see the resultls on `/outputs' and for more options, see opts.py.

## 2.2 NSC fine tuning
After 2.1, please rename your outputs file to prepare reinforcement learning.
`M9` -> 'M9_rl'
`infos_M9.pkl` -> `infos_M9_rl.pkl`
`infos_M9-best.pkl` -> `infos_M9_rl-best.pkl`

```bash
$ python tools/train.py --cfg configs/fc_rl.yml --id fc_rl
```

# Reference
This codebase is built upon the official implementation of the following. Consider citing their work if you find this repo useful.
```
@article{luo2018discriminability,
  title={Discriminability objective for training descriptive captions},
  author={Luo, Ruotian and Price, Brian and Cohen, Scott and Shakhnarovich, Gregory},
  journal={arXiv preprint arXiv:1803.04376},
  year={2018}
}
```
```
@inproceedings{kuo2022pretrained,
    title={Beyond a Pre-Trained Object Detector: Cross-Modal Textual and Visual Context for Image Captioning},
    author={Chia-Wen Kuo and Zsolt Kira},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2022}
}
```
```
@inproceedings{cornia2020m2,
    title={{Meshed-Memory Transformer for Image Captioning}},
    author={Cornia, Marcella and Stefanini, Matteo and Baraldi, Lorenzo and Cucchiara, Rita},
    booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    year={2020}
}
```
