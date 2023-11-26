# To-Read-More
The Code and Data of the paper To Read More.

# 1. Preprocess data of MS-COCO
Download preprocessed coco captions from [link](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) from Karpathy's homepage. Extract `dataset_coco.json` from the zip file and copy it in to `data/`. This file provides preprocessed captions and also standard train-val-test splits.

Then do:

```bash
$ python scripts/prepro_labels.py --input_json data/dataset_coco.json --output_json data/cocotalk.json --output_h5 data/cocotalk
```

`prepro_labels.py` will map all words that occur <= 5 times to a special `UNK` token, and create a vocabulary for all the remaining words. The image information and vocabulary are dumped into `data/cocotalk.json` and discretized caption data are dumped into `data/cocotalk_label.h5`.

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
