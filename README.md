# PENCIL.pytorch
PyTorch implementation of [Probabilistic End-to-end Noise Correction for Learning with Noisy Labels](https://arxiv.org/abs/1903.07788), CVPR 2019.

## Requirements:
+ python3.6
+ numpy
+ torch-0.4.1
+ torchvision-0.2.0

## Usage
+ On CIFAR-10, we retained 10% of the CIFAR-10 training data as the validation set and modify the original correct labels to obtain different noisy label datasets.
+ So the validation set is part of `data_batch_5`, and both of them have 5000 samples  
+ Add symmetric noise on CIFAR-10: `python addnoise_SN.py`
+ Add asymmetric noise on CIFAR-10: `python addnoise_AN.py`
+ `PENCIL.py` is used for both training a model on dataset with noisy labels and validating it

## options
+ `b`: batch size
+ `lr`: initial learning rate of stage1
+ `lr2`: initial learning rate of stage3
+ `alpha`: the coefficient of Compatibility Loss
+ `beta`: the coefficient of Entropy Loss
+ `lambda1`: the value of lambda
+ `stage1`: number of epochs utill the end of stage1
+ `stage2`: number of epochs utill the end of stage2
+ `epoch`: number of total epochs to run
+ `datanum`: number of train dataset samples
+ `classnum`: number of train dataset classes

## The framework of PENCIL

![framework.eps](https://github.com/yikun2019/PENCIL/blob/master/framework.eps)
## The proportion of correct labels on CIFAR-10
![SN30.eps](https://github.com/yikun2019/PENCIL/blob/master/SN70.eps)
![AN30.eps](https://github.com/yikun2019/PENCIL/blob/master/AN30.eps)
## The results on real-world dataset Clothing1M
 |method|Test Accuracy (%)
---|:--:|:---:
1|Cross Entropy Loss|68.94
2|Forward [1]|69.84
3|Tanaka *et al*. [2]|72.16
4|PENCIL|**73.49**
## Citing this repository
If you find this code useful in your research, please consider citing us:

```
@inproceedings{ML_GCN_CVPR_2019,
author = {Kun, Yi and Jianxin, Wu},
title = {{Probabilistic End-to-end Noise Correction for Learning with Noisy Labels}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
```
## Reference
[1] Giorgio Patrini, Alessandro Rozza, Aditya Krishna Menon, Richard Nock, and Lizhen Qu. [Making deep neural networks robust to label noise: A loss correction approach](http://arxiv.org/abs/1609.03683). In CVPR, pages 1944–1952, 2017.
</br>[2] Daiki Tanaka, Daiki Ikami, Toshihiko Yamasaki, and Kiyoharu Aizawa. [Joint optimization framework for learning with noisy labels](https://arxiv.org/abs/1803.11364). In CVPR, pages 5552–5560, 2018.