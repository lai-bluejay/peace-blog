---
title: '深度学习课程笔记：fastai course-v3 lesson1 '
date: '2019-03-12 22:54:00'
updated: '2019-03-23 12:02:07'
tags:
  - fastai
  - 深度学习
  - 课程笔记
categories:
  - 深度学习
abbrlink: c8c4f3aa
---
fast.ai course-v3 笔记

  

## fast.ai 介绍

fastai库可以说是深度学习包pytorch最佳实践，简化了快速准确的神经网络训练，几乎做到“开箱即用”的支持vision，text，tabular，和collab（协同过滤）的模型，实现了常见的模型结构，诸如resnet34，resnet50等。基本可用完成10行代码进行模型训练、评估等，非常适合快速上手和深度学习入门

  

## 代码结构
![fastai代码结构 by lai7bluejay 深度学习](https://img.hacpai.com/file/2019/03/dep1-3c0140e9.jpg)



# pre-lesson

  

## 在哪儿进行fast.ai course-v3的学习呢

  

[course-v3](https://github.com/fastai/course-v3/tree/master/nbs/dl1) 主体部分是jupyter notebook，因此，需要一台有GPU+jupyter的服务器。

  

但很多数据集和预训练的模型都在aws或者Dropbox上，如果使用国内的服务器，没有高带宽的ss是很吃亏的。之前在公司的双卡服务器上进行了尝试，1m/s的网速，1080Ti的训练速度，进行lesson1 简直欲仙欲死。

  

因此推荐几个官方解决方案，在数据获取和模型训练速度上，拥有质的变化。

  

### CoLab

  

**优点**：网速飞快，训练飞快

  

**缺点**：墙，需要梯子。

  

1. 登录Colab[点击](https://colab.research.google.com/), 点击GITHUB选项卡，输入`fastai/course-v3`，就能加载当前项目的ipynb。

2. 修改运行时环境：点击runtime，选择GPU，save。

3. 运行` !curl -s https://course.fast.ai/setup/colab | bash`，初始化notebook

4. 疯狂点确定。存储备份在自己的drive上。

  

官网参考[here](https://course.fast.ai/start_colab.html)。

  
  

### Crestle.ai

  
  

参考[crestle.ai](https://course.fast.ai/start_crestle.html)

  

没使用过

  

### Kaggle Kernels

  

优点：只要有kaggle账户就能直接使用，没被墙

  

缺点：访问速度一般~~；训练速度较慢，K80，据说resnet34 40分钟

  

参考：[kaggle kernels](https://course.fast.ai/start_kaggle.html)

  

# lesson 1 What's your pet

  

图像分类初级课程，快速构建resnet34的图片分类器。

  

## 课程设置

·该课程由Jeremy Howard和Rachel Thomas教授。

  

·本课程的先决条件是“高中数学+ 1年编码经验”。Python编码知识是一个优势。

  

·该课程包括7节课，每节课约1.5至2小时，每周工作8-10小时。建议您通过完整的课程/视频，而不是尝试理解/谷歌一切。

  

### 数据集说明

  

[Oxford-IIIT Pet Dataset](http://www.robots.ox.ac.uk/~vgg/data/pets/)，2012年的分类准确率59.21%。

  

37个不同的宠物类别。每个类约有200张图像。它有12只猫和25种犬种。本课程的早期版本使用了“猫与狗”数据集，这些数据集多年来使用易于获得的模型变得过于容易。Oxford-IIIT宠物数据集分类是细粒度分类的问题（即类似类别之间的分类）。

  

·**标签是预测过程的目标**。在数据集中，各个图像的名称格式为：'path_label_extension.jpg'。正则表达式用于从文本中提取标签。

  

·**图像大小**：在本课程中，使用的图像是方形（即高度=宽度），大小为224 x 224.不同形状/大小的图像会根据大小调整大小并进行裁剪。深度学习模型的缺点之一是它们需要相同大小的图像。可变尺寸图像在课程的第2部分的范围内。

  

·**databunch对象**：包含培训和验证数据

  

·**图像的标准化**：单个像素值的范围从0到255.它们被标准化以使值为平均值0和标准差1.图像增强：从现有图像生成新图像。有助于避免过度贴合。填充：将在后续课程中讨论的概念。
  

### notebook说明

  

#### untar_data

  

把url的数据fname下载并解压到dest。

  

####  ImageDataBunch

  

fastai定义的图片数据结构

  

## 模型

CNN（卷积神经网络）需要train2件事：数据和模型结构

  
  

#### resnet34

  

cnn中有一些在很多任务上表现很好很稳定的模型结构，例如resnet34，resnet50，我们可以直接使用预定义好的resnet34的模型，进行快速的模型训练。

  

调用的时候会使用在imagenet预训练好权重的resnet34，进行fine-tune。预训练好的模型是针对1000+类图片进行分类，因此本来就对pet数据会有一定的效果。 resnet50也是很好的选择，但需要更多的计算能力。更多的图像模型效果参考Stanfor的[benchmark](https://dawn.cs.stanford.edu/benchmark/)

 
> **为什么选择resnet34**
因为在图片相关的训练任务重，resnet34，resnet50都取得了几乎最好的结果（参考stanford 的dawn benchmark）。而对于深度学习图像分类而言，从0开始训练图像分类，不如从已有pre-trained模型进行fine-tune得到的结果更好、更高效。Know something to anything, instead of knows nothing.
  

·** ResNet vs Inception:** resnet获得了更好的结果。

  

**Lower layers vs higher layers**

- 更低层：更多的是基础的模式。比如第一层一般是水平线、垂线、对角线等

- 第二层是把第一层的特征结合起来

- 由于基本的模式或者几何图形大致相同，因此无需大量训练低层模型，也给模型的迁移提供了基础

- 与未训练的模型相比，unfreeze所有层会导致准确性的丢失，因为底层的权重收到了干扰。不同的层也给出了不同的语义复杂度。因此更早的层应该以更低的学习率fine tune或者不训练

- 可以查看这篇文章，观察每层cnn获取的特征[Zeiler and Fergus paper on visualizing CNN](https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)

- lr_finder. 传入不同的学习率，可以查看参数更新的速率。



调用的时候会使用在imagenet预训练好权重的resnet34，进行fine-tune。预训练好的模型是针对1000+类图片进行分类，因此本来就对pet数据会有一定的效果。

  

验证集验证，error_rate评估_

  

#### fit_one_cycle
![cnn fit4cycle的结果 by lai7bluejay 深度学习](https://img.hacpai.com/file/2019/03/fit4cycle-300bf1e4.png)

之后会有文章专门讲fit one cycle 这件事。

老师在课程中说到，尽管去跑代码，知道你要做的任务的输入是什么，输出什么样的结果。多跑了代码，就会慢慢开始对自己要做的事情有所了解。如果你花70个小时沉浸在书里，最终获得的收益反而不如先上手deep learning，再补充理论基础。

runing code, knows what goes in and what comes out. Run the code, you'd know how it works out.

  

suggest a potential breakthrough, with more robust language models can help researchers tackle a range of unsolved problems.

  

### 参考网站

  

https://dawn.cs.stanford.edu/benchmark/