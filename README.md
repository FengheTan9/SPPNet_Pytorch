# SPPNet Pytorch

# SPPNET描述

SPPNET是何凯明等人2015年提出。该网络在最后一层卷积后加入了空间金字塔池化层(Spatial Pyramid Pooling layer)替换原来的池化层(Pooling layer),使网络接受不同的尺寸的feature maps并输出相同大小的feature maps，从而解决了Resize导致图片型变的问题。

[论文](https://arxiv.org/pdf/1406.4729.pdf)： K. He, X. Zhang, S. Ren and J. Sun, "Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 37, no. 9, pp. 1904-1916, 1 Sept. 2015, doi: 10.1109/TPAMI.2015.2389824.

# 模型架构

SPPNET基于ZFNET，ZFNET由5个卷积层和3个全连接层组成，SPPNET在原来的ZFNET的conv5之后加入了Spatial Pyramid Pooling layer。

# 数据集

使用的数据集：[ImageNet2012](http://www.image-net.org/)

- 数据集大小：共1000个类、224*224彩色图像
    - 训练集：共1,281,167张图像
    - 测试集：共50,000张图像

- 数据格式：JPEG
    - 注：数据在dataset.py中处理。

- 下载数据集。目录结构如下：

```text
└─dataset
    ├─ilsvrc                # 训练数据集
    └─validation_preprocess # 评估数据集
```

# 环境要求

- Pytorch

# 运行

单卡：

```python
# 单卡ZFNet
python train_zfnet_1.py
# 单卡SPPNet(Single training mode)
python train_single_1.py
# 单卡SPPNet(Multi training mode)
python train_mult_1.py
```

多卡：

```python
# 多卡ZFNet
python train_zfnet_8.py
# 多卡SPPNet(Single training mode)
python train_single_8.py
# 多卡卡SPPNet(Multi training mode)
python train_mult_8.py
```

# 脚本说明和参数

## 脚本说明

```bash
├── README.md                                // SPPNet相关说明
├── src
   ├──dataset.py                             // 创建数据集
   ├──sppnet.py                              // sppnet架构
   ├──zfnet.py                               // zfnet架构
   ├──spatial_pyramid_pooling.py             // 金字塔池化层架构
├── train_zfnet_1.py                         // 训练脚本
├── train_single_1.py                        // 训练脚本
├── train_mult_1.py                          // 训练脚本
├── train_zfnet_8.py                         // 训练脚本
├── train_single_8.py                        // 训练脚本
├── train_mult_8.py                          // 训练脚本
```
## 脚本参数

在config.py中可以同时配置训练参数和评估参数：

  ```bash
  # zfnet配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 150,              # epoch大小
  'batch_size': 256,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'T_max': 150,                   # 余弦退火最大迭代次数
  'lr_init': 0.035,               # 初始学习率
  'weight_decay': 0.0001,         # 权重衰减

  # sppnet(single train)配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 160,              # epoch大小
  'batch_size': 256,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'T_max': 150,                   # 余弦退火最大迭代次数
  'lr_init': 0.01,                # 初始学习率
  'weight_decay': 0.0001,         # 权重衰减


  # sppnet(mult train)配置参数
  'num_classes': 1000,            # 数据集类别数量
  'momentum': 0.9,                # 动量
  'epoch_size': 160,              # epoch大小
  'batch_size': 128,              # 输入张量的批次大小
  'image_height': 224,            # 图片长度
  'image_width': 224,             # 图片宽度
  'T_max': 150,                   # 余弦退火最大迭代次数
  'lr_init': 0.01,                # 初始学习率
  'weight_decay': 0.0001,         # 权重衰减
  ```
