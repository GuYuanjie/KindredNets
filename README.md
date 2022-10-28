# HorizonFlow Lab
Welcome new members who has strong insight and motivition in AI and CS cross-fields. Feel free to email your CV/resume to our PI
@ [Yuanjie Gu](https://github.com/GuYuanjie).
Email1: yuanjie_gu@163.com
Email2: yuanjie@horizonflow.top

# KindredNets (Deep Low-Excitation Fluorescence Imaging Enhancement)

By YUANJIE GU,† ZHIBO XIAO,† WEI HOU, CHENG LIU, YING JIN, AND SHOUYU WANG*

This work is submitted to arXiv and Optica. Good luck to us.
This repo provides simple testing codes, pretrained models and the network strategy demo.



## BibTex

```
@article{Gu:22,
author = {Yuanjie Gu and Zhibo Xiao and Wei Hou and Cheng Liu and Ying Jin and Shouyu Wang},
journal = {Opt. Lett.},
keywords = {Confocal laser scanning microscopy; Fluorescence; Image enhancement; Image processing; Neural networks; Nonlinear microscopy},
number = {16},
pages = {4175--4178},
publisher = {Optica Publishing Group},
title = {Deep low-excitation fluorescence imaging enhancement},
volume = {47},
month = {Aug},
year = {2022},
url = {https://opg.optica.org/ol/abstract.cfm?URI=ol-47-16-4175},
doi = {10.1364/OL.466050},
abstract = {In this work, to the best of our knowledge, we provide the first deep low-excitation fluorescence imaging enhancement solution to reconstruct optimized-excitation fluorescence images from captured low-excitation ones aimed at reducing photobleaching and phototoxicity due to strong excitation. In such a solution, a new framework named Kindred-Nets is designed aimed at improving the effective feature utilization rate; and additionally, a mixed fine-tuning tactic is employed to significantly reduce the required number of fluorescence images for training but still to increase the effective feature density. Proved in applications, the proposed solution can obtain optimized-excitation fluorescence images in high contrast and avoid the dimming effect due to negative optimization from the ineffective features on the neural networks. This work can be employed in fluorescence imaging with reduced excitation as well as extended to nonlinear optical microscopy especially in conditions with low output nonlinear signals. Furthermore, this work is open source available at https://github.com/GuYuanjie/KindredNets.},
}
```

## Complete Architecture

The complete framework of KindredNets is shown as follows,

![](figures/architecture.jpg)

# Implementation

## This work was inspired by the following work
```
@ARTICLE{DLN2020,
  author={Li-Wen Wang and Zhi-Song Liu and Wan-Chi Siu and Daniel P.K. Lun},
  journal={IEEE Transactions on Image Processing}, 
  title={Lightening Network for Low-light Image Enhancement}, 
  year={2020},
  doi={10.1109/TIP.2020.3008396},
}
```
https://github.com/WangLiwen1994/DLN
thanks！！！
## Prerequisites

- Python 3.5
- NVIDIA GPU + CUDA
- [optional] [sacred+ mongodb (experiment control)](https://pypi.org/project/sacred/) 

## Getting Started

### Installation

- Install PyTorch and dependencies from http://pytorch.org
- Install python libraries:

```bash
pip install pillow, opencv-python, scikit-image, sacred, pymongo
```

- Clone this repo


### Testing

- A few example test images are included in the `./test_img` folder.
- Please use trained model`./Lighten_pretrained_best_psnr2140-ssim8001.pth` and `Darken_pretrained_best_psnr2140-ssim8001.pth`
  - Put them under `./models/`
- Test the model by:

```bash
python test.py --modelfile models/PLN_pretrained.pth

# or if the task towards image enhancement with mixed fine tune.
python test.py --modelfile models/PLN_CFFI_LOL.pth
```

The test results will be saved to the folder: `./output`.


### Dataset

- Download the VOC2007 dataset and put it to "datasets/VOC2007/".
- Download the LOL dataset and put it to "datasets/LOL".
- We provide a valuable confocal fluorescence microscopy dataset CFFI in "datasets/CFFI".
- Mixed fine tune dataset is combined whit the LOL dataset and our CFFI dataset.

### Training

It needs to manually switch the training dataset: 

1) first, train from the synthesized dataset, 
2) then, load the pretrained model and train with mixed fine tune dataset.

```bash
python train.py 
```

### Then run test to applications.
