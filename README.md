基于层级特征交互与增强感受野的双分支遥感图像去雾网络
==
Created by Shuailing Fang, [Hang Sun](https://github.com/sunhang1986), Zhiping Dan from Department of Computer and Information, China Three Gorges University.

Introduction
--
 本文提出了基于层级特征交互与增强感受野的双分支遥感图像去雾算法，该方法包含层级特征交互子网和多尺度信息提取子网。
--
Prerequisites
+ Pytorch 1.7.1
+ Python 3.6.12
+ CUDA 8.0
+ Ubuntu 18.04

Test
--
The [Download](https://www.dropbox.com/s/k2i3p7puuwl2g59/Haze1k.zip?dl=0) path of haze1k dataset . the [Download](https://github.com/BUPTLdy/RICE_DATASET.) path of RICE dataset . 

Test the model:

 python   test.py (You need to specify the test data directory and the pre-training model directory in the test.py file)

## Citation


```
@InProceedings{Yu_2021_CVPR,
    author    = {Yu, Yankun and Liu, Huan and Fu, Minghan and Chen, Jun and Wang, Xiyao and Wang, Keyan},
    title     = {A Two-Branch Neural Network for Non-Homogeneous Dehazing via Ensemble Learning},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {193-202}
}
```



>>>>>>> 66728f3 (first)
>>>>>>> b08b827 (123)
