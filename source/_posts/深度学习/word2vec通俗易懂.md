---
title: word2vec ，通俗易懂
date: '2019-05-08 17:19:14'
updated: '2019-05-09 10:53:51'
tags:
  - word2vec
  - 词嵌入
  - nlp
mathjax: true
categories:
  - 深度学习
abbrlink: 13a5ed49
---
word2vec

## 0. 应该阅读的参考文献

1. Mikolov 两篇原论文：
    - Distributed Representations of Sentences and Documents：word2vec框架
    - Efficient estimation of word representations in vector space：训练w2v的两个trick，hierarchical softmax和negative sampling
2. [深度学习word2vec笔记之基础篇](https://link.jianshu.com/?t=http%3A%2F%2Fblog.csdn.net%2Fmytestmy%2Farticle%2Fdetails%2F26961315)
3. Xin Rong 的论文：『word2vec Parameter Learning Explained』



### 0.1 词的向量表示

#### One-hot  representation

即在每个词使用一个长向量表示，向量长度是词典大小，词对应位置的分量为1，其余为0。

优点：简单易懂；稀疏表达也相对简洁

缺点：词典过大导致维数灾难；稀疏计算量大；词与词之间没有任何语义关系，损失语义信息；复用性极低



#### distributed representation

Hinton于1986年提出，基本思想是通过训练，使用一个普通向量表示一个词语。

优点：维度可控；向量空间有物理含义，可以表示语义相似性；

缺点：训练。



## 1. 语言模型

简单理解：给定一个x $\in$ 文档R，求上下文y的概率。模型$f(x) \rightarrow y$.

数学形式上可以表示为:  给定T个词的字符串s，符合自然语言习惯的概率是：
$$
\mathrm{p}(s)=\mathrm{p}\left(w_{1}, w_{2}, \cdots, w_{T}\right)=\mathrm{p}\left(w_{1}\right) \mathrm{p}\left(w_{2} | w_{1}\right) \mathrm{p}\left(w_{3} | w_{1}, w_{2}\right) \cdot \cdot \mathrm{p}\left(w_{t} | w_{1}, w_{2}, \cdots w_{T-1}\right)
$$
如果一句话出现的概率低，则认为不符合语言习惯。也可以写成:
$$
\mathrm{p}(\mathrm{s})=\mathrm{p}\left(w_{1}, w_{2}, \cdots, w_{T}\right)=\prod_{i=1}^{T} p\left(w_{i} | \text {Context}_{i}\right)
$$
那么，计算一句话是否符合语言习惯的任务就落到如何计算$p(w_i|Context_i)$ 上，以及缩减$Context_i$的范围 （剪枝）。



### 1.1       N-gram

当计算p时，如果词表越大，可能出现的上下文组合就越多，p的计算量就会越多。考虑到语言学现象，大部份词都是以词组的形式出现，也就是影响一个词的出现概率，往往是上文的几个词决定的，因此有了n-gram，表达如下。
$$
\mathrm{p}\left(w_{i} | Context_{i}\right)=\mathrm{p}\left(w_{i} | w_{i-n+1}, w_{i-n+2}, \cdots, w_{i-1}\right)
$$


优点：

缺点：n的数量决定了最终的效果和计算量；对于相同上文的词，得出的概率是不同的，无法表示语义相似度；对于某些n-gram如果语料中没有出现，会导致词序列p=0，需要加入拉普拉斯平滑、或者使用n-1 gram代替n-gram

当然，也有把上文词映射到词性上，判断若干词性之后的词的概率，减少计算空间。

### 1.2 词嵌入

基于分布假设，即出现在类似语境中的词语具有相似含义。早期使用向量空间模型在连续向量空间中表示词语，“相似的词在彼此附近嵌入”，指相似的词映射到附近的点。一般有两类方法：基于计数的方法（如LDA）；预测方法（如nnlm，神经网络语言模型）。

两类方法的区别：

> 简而言之：基于计数的方法会计算在大型文本语料库中，一些字词与临近字词共同出现的频率统计数据，然后将这些计数统计数据向下映射到每个字词的小型密集向量。预测模型会根据学到的小型密集嵌入向量（被视为模型的参数），直接尝试预测临近的字词。

word-embeding的优势如下：

1. 将词语映射到固定长度的小型密集嵌入向量，可计算
2. 词向量本身具备语义信息
3. 可复用成为可能。

word2vec属于预测模型，计算效率极高。



## 2. word2vec的训练模式



### 2.0 简述

如果不想看公式推导，可以跳过剩下的内容，直接到第4小节。

在word2vec中，有两个基本结构：skip-gram 和CBOW。假设现在我们要得到V个单词的，分量大小为 N 词向量H。

- skip-gram：给定中心词，预测上下文

- CBOW：给定上下文，预测词语本身

简单来说，训练出来的词向量即神经网络的权重，那为什么会有V个词向量呢：

- 由于初始向量$w_k$只有第k个分量为1，使得最终每个词只有权重矩阵的第k行向量$h_k$表示
- 由于初始的one-hot每次词的位置是不同的，因此每个词的向量表示就是唯一的。

也就是说，最初的单词的向量大小由 V  变成了隐层大小N，同时完成了词嵌入表示和降维，使得计算成为可能。

**其实关于word2vec，最重要的就是记住这两点，记住了就能直观理解word2vec所发生的事情**



用浅显的语言总结一下CBOW的过程：

**前向传播**：多个上下文词向量和权重矩阵W 的乘机平均值作为隐层h，在用隐层乘矩阵W’ ，进行softmax之后得到待预测词的向量y’。

**损失函数**：需要极大化目标单词的出现概率（似然），即极小化复对数似然函数，得到Loss。

**反向传播**：根据Loss进行梯度下降，更新权重。



用浅显的语言总结一下skip-gram的过程：

**前向传播**：将输入词向量和权重矩阵W 的乘积作为隐层h（相当于取出了对应的权重向量）；对于每个输出单元$y_i$, 使用$h_i$在用隐层乘矩阵W’ ，进行softmax之后得到待预测词的向量y’。

需要注意的是，从隐层到输出层的计算过程中，权重矩阵$W^{\prime}$是共享的，但对于每个输出单元来说，计算的loss的值是不一致的。虽然看似不够精细，但训练这样一个词向量也是我们所需要的。

**损失函数**：需要极大化目标单词的出现概率（似然），即极小化复对数似然函数，得到Loss。

**反向传播**：根据Loss进行梯度下降，更新权重。



我们可以看到，在更新权重的过程中，需要计算所有v 和v’，更新整个矩阵，计算量还是很大的。



原始的方法所存在的问题是计算量太大，体现在以下两方面：

1. 前向过程，h->y这部分在对向量进行softmax的时候，需要计算V次。
2. 后向过程，softmax涉及到了V列向量，所以也需要更新V个向量。

问题就出在V太大，而softmax需要进行V次操作，用整个W进行计算。

因此word2vec使用了两种优化方法，Hierarchical SoftMax和Negative Sampling，对softmax进行优化，不去计算整个W，大大提高了训练速度。

简单说明一下HS和NS：

- Hierachical Softmax：把V分类问题变成log(V)次的二分类。
- Negative sampling：计算总问题的一个子集

记住，这些都不是word2vec独有的优化训练trick。Hierachical softmax是计算softmax的一个有效方案，是通用的。而negative sampling则是更直截了当地只更新一部分权重矩阵W'的向量





### 2.1  Continuous Bag-of-Word Model 

CBOW, 连续词袋模型，给定上下文的词序列，预测词本身。

#### 2.1.1 one word context

CBOW的最简单模式，假设给定一个context词语，就能预测词。假设词表大小V

![img](https://pic4.zhimg.com/80/v2-a1a73c063b32036429fbd8f1ef59034b_hd.jpg)

输入:  $X = [\vec x\_1, \cdots, \vec x\_V]$ , 隐层$H\_N = X\_{V} W\_{V \times N}$。其中$\vec x_i$是原始词的onehot，隐层全连接。

权重矩阵W如下：

$$W=\left( \begin{array}{cccc}{\omega_{11}} & {\omega_{12}} & {\dots} & {\omega_{1 N}} \\\\ {\omega_{21}} & {\omega_{22}} & {\dots} & {\omega_{2 N}} \\ {\cdots} & {\cdots} & {\cdots} & {\cdots} \\\\ {\omega_{V 1}} & {\omega_{V 2}} & {\dots} & {\omega_{V N}}\end{array}\right)$$

每一个行向量表示每个词的的n维向量$v_\omega$。假定给定context输入词$\omega_k$, 只有第k个分量为1，则:

$$\mathbf{h_k}=\mathbf{W}^{T} x=\mathbf{W}_{(k, *)}^{T} x_{k}=\mathbf{v}_{\omega_{k}}^{T} =\sum_{k=1}^{V} x_{k} \cdot \omega_{k i}$$


也就是h向量会等于权值矩阵的第k行，长度为1 x N。同样对于隐层 -> 输出层的矩阵$\mathbf{W}^{\prime}\_{N \times V}=\{\omega\_{i j}^{\prime}\}$。

我们注意看这个公式，结合我们的目标来理解这个公式所表达的含义：

- 计算结果：网络权重。
- 由于向量$w_k$只有第k个分量为1，使得最终每个词只有权重矩阵的第k行向量$h_k$表示
- 由于初始的one-hot每次词的位置是不同的，因此每个词的向量表示就是唯一的。

也就是说，最初的单词的向量大小由 V  变成了隐层大小N，同时完成了词嵌入表示和降维，使得计算成为可能。

最终每个单词的得分，由列向量$\omega\_{ij} \cdot h\_k, \omega\_{ij} 大小N \times 1$, 即：
$$
\mu_{j}=\mathbf{v}_{\omega_{\mathrm{j}}}^{\prime \  T} \mathbf{h}
$$

对于输入单词$w_k$,  使用softmax计算单词分量$w_j$的后验分布为：

$$p\left(\omega_{j} | \omega_{k}\right)=y_{j}=\frac{\exp \left(\mu_{j}\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(\mu_{j^{\prime}}\right)}$$

将公式(5) 和 (6) 带入，得到我们的目标函数 max p。

$$\begin{array}{l}{\max p\left(\omega_{O} | \omega_{I}\right)=\max \ y_{j} *} \\ {=\max \ \log y_{j}^{*}} \\ {=\mu_{j^{*}}-\log \sum_{j^{\prime}=1}^{V} \exp \left(\mu_{j^{\prime}}\right) :=-E}\end{array}$$

现在，我的目标函数变成了最小化loss ： $E = p\left(\omega_O | \omega_I\right)$





####  2.1.2 Multi-Context

对于多个上下文的CBOW，隐层的计算由1个context向量变成了C 个context单词向量的平均值。
$$
\begin{aligned} \mathbf{h} &=\frac{1}{C} \mathbf{W}^{T}\left(\mathbf{x}_{1}+\mathbf{x}_{2}+\cdots+\mathbf{x}_{C}\right) \\ &=\frac{1}{C}\left(\mathbf{v}_{\omega_{1}}+\mathbf{v}_{\omega_{2}}+\cdots+\mathbf{v}_{\omega C}\right)^{T} \end{aligned}
$$

我们通过对multi-context的简单推导来了解CBOW的权重更新方式和优缺点。

$$
\begin{array}{l}{E=-\log p\left(\omega_{O} | \omega_{I, 1}, \ldots, \omega_{I, C}\right)} \\ {=-\mu_{j^{*}}+\log \sum_{j^{\prime}=1}^{V} \exp \left(\mu_{j^{\prime}}\right)} \\ {=-\mathbf{v}_{\omega O}^{\prime T} \cdot \mathbf{h}+\log \sum_{j^{\prime}=1}^{V} \exp \left(\mathbf{v}_{\omega j}^{\prime T} \cdot \mathbf{h}\right)}\end{array}
$$

我们分别来看，隐层-> 输出层，和输入-> 隐层的权重更新方式。

首先，对得分求偏导
$$
\frac{\partial E}{\partial \mu_{j}}=y_{j}-t_{j} :=e_{j} \\
\frac{\partial E}{\partial \omega_{i j}^{\prime}}=\frac{\partial E}{\partial \mu_{j}} \cdot \frac{\partial \mu_{j}}{\partial \omega_{i j}^{\prime}}=e_{j} \cdot h_{i}
$$

使用SGD进行权重更新的话，新的权重向量更新为:

$$
\mathbf{v}_{\omega_{j}}^{\prime(n e w)}=\mathbf{v}_{\omega_{j}}^{\prime(o l d)}-\eta \cdot e_{j} \cdot \mathbf{h} \text { for } j=1,2, \ldots V
$$

其中，$\eta$为学习率。$t_j = 1， 当且仅当输出单元等于真实单词的时候成立$

$$
t_j = \begin{cases}
\ 1, \ if \ \ j = j^* \\\\
\ 0
\end{cases}
$$


对于输入层到隐层：

首先，对隐层求偏导:
$$
\frac{\partial E}{\partial h_{i}}=\sum_{j=1}^{V} \frac{\partial E}{\partial \mu_{j}} \cdot \frac{\partial \mu_{j}}{\partial h_{i}}=\sum_{j=1}^{V} e_{j} \cdot \omega_{i j}^{\prime} :=E H_{i} \\
\frac{\partial E}{\partial \omega_{k i}}=\frac{\partial E}{\partial h_{i}} \cdot \frac{\partial h_{i}}{\partial \omega_{k i}}=E H_{i} \cdot x_{k}
$$

则输入的权重矩阵的更新为:

$$
\mathbf{v}_{\omega I, c}^{(n e w)}=\mathbf{v}_{\omega I, c}^{(o l d)}-\frac{1}{C} \cdot \eta \cdot E H^{T} \text { for } c=1,2, \ldots, C
$$

通过公式 我们可以看到，梯度下降使得$\sum e_j$ 越来越小。



#### 2.1.3 浅显总结

用浅显的语言总结一下CBOW的过程：

**前向传播**：多个上下文词向量和权重矩阵W 的乘机平均值作为隐层h，在用隐层乘矩阵W’ ，进行softmax之后得到待预测词的向量y’。

**损失函数**：需要极大化目标单词的出现概率（似然），即极小化复对数似然函数，得到Loss。

**反向传播**：根据Loss进行梯度下降，更新权重。



我们可以看到，在更新权重的过程中，需要计算所有v 和v’，更新整个矩阵，计算量还是很大的。



### 2.2 Skip-gram

skip-gram是通过给定中心词，来预测上下文的概率。在skip-gram中，隐层的计算相对与单个上下文的CBOW没有变化，但在输出层中，多了C个输出单元的多项式分布。

$$
p\left(\omega_{c, j}=\omega_{O, c} | \omega_{I}\right)=y_{c, j}=\frac{\exp \left(\mu_{c, j}\right)}{\sum_{j^{\prime}=1}^{V_{j}} \exp \left(\mu_{j}^{\prime}\right)}
$$

换句话说，理解单个上下文的CBOW，和单个上下文的skip-gram是一致的。$\boldsymbol{\omega}_{O, c}$表示的是输出上下文单词（output context words）的第c个单词。

> 插一句
>
> 在skip-gram中的输出层有个panel的概念，就是输出层的表示每个上下文单词的神经元的组合。图中一种有C个context words，所以总共有C个panel

$\mu _{i, j} $表示的是输出层第c个panel的第j个神经元的输入值；**输出层的所有panels共享同一权重矩阵W′**，

$$\mu_{c, j}=\mu_{j}=\mathbf{v}_{\omega_{j}}^{\prime T} \cdot \mathbf{h}, \text { for } c=1,2, \ldots, C$$

v’也是W’中的列向量。损失函数为最小负对数似然：

$$\begin{array}{l}{E=-\log p\left(\omega_{O, 1}, \omega_{O, 2}, \ldots, \omega_{O, C} | \omega_{I}\right)} \\ {=-\log \prod_{c=1}^{C} \frac{\exp \left(\mu_{c, j_{c}} *\right)}{\sum_{j^{\prime}=1}^{V} \exp \left(\mu_{j^{\prime}}\right)}} \\ {=-\sum_{c=1}^{C} \mu_{j_{c}}^{*}+C \cdot \log \sum_{j^{\prime}=1}^{V} \exp \left(\mu_{j^{\prime}}\right)}\end{array}$$

利用链式法则求偏导：
$$
\frac{\partial E}{\partial \mu_{c, j}}=y_{c, j}-t_{c, j} :=e_{c, j}
$$

同样，我们定义V维的向量EI为所有上下文单词的预测误差之和，

$$
E I_{j}=\sum_{c=1}^{C} e_{c, j}
$$

则关于$\omega^{\prime}_{i,j}$的偏导:

$$
\frac{\partial E}{\partial \omega_{i j}^{\prime}}=\sum_{c=1}^{C} \frac{\partial E}{\partial \mu_{c, j}} \cdot \frac{\partial \mu_{c, j}}{\partial \omega_{i j}^{\prime}}=E I_{j} \cdot h_{i}
$$

hidden -> output的权重向量更新为:

$$
\mathbf{v}_{\omega_{j}}^{\prime(n e w)}=\mathbf{v}_{\omega_{j}}^{\prime(o l d)}-\eta \cdot E I_{j} \cdot \mathbf{h} \text { for } j=1,2, \ldots, V
$$

输出层的预测误差的计算是基于多个上下文单词context words,而不是单个目标单词 target word;需注意的是对于每一个训练样本，我们都要利用该参数更新公式来更新hidden→output权重矩阵**W**′的每个元素。

对于输入层到隐层，需要替换的是误差变成了多个上下文，即$e_j$变成了 $EI_j$。权重更新为:

$$
\mathbf{v}_{\omega_{I}}^{(n e w)}=\mathbf{v}_{\omega_{I}}^{(o l d)}-\eta \cdot E H^{T}
$$


#### 浅显总结

同样，我们做一个简单的总结。

用浅显的语言总结一下skip-gram的过程：

**前向传播**：将输入词向量和权重矩阵W 的乘积作为隐层h（相当于取出了对应的权重向量）；对于每个输出单元$y_i$, 使用$h_i$在用隐层乘矩阵W’ ，进行softmax之后得到待预测词的向量y’。

需要注意的是，从隐层到输出层的计算过程中，权重矩阵$W^{\prime}$是共享的，但对于每个输出单元来说，计算的loss的值是不一致的。虽然看似不够精细，但训练这样一个词向量也是我们所需要的。

**损失函数**：需要极大化目标单词的出现概率（似然），即极小化复对数似然函数，得到Loss。

**反向传播**：根据Loss进行梯度下降，更新权重。



我们可以看到，在更新权重的过程中，需要计算所有v 和v’，更新整个矩阵，计算量还是很大的。



### 2.3 小结

原始的方法所存在的问题是计算量太大，体现在以下两方面：

1. 前向过程，h->y这部分在对向量进行softmax的时候，需要计算V次。
2. 后向过程，softmax涉及到了V列向量，所以也需要更新V个向量。

问题就出在V太大，而softmax需要进行V次操作，用整个W进行计算。

因此word2vec使用了两种优化方法，Hierarchical SoftMax和Negative Sampling，对softmax进行优化，不去计算整个W，大大提高了训练速度。



## 3. 训练技巧

实际上，Hierarchical SoftMax和Negative Sampling都属于计算过程中的优化方法。对于输出V个词的多分类问题，softmax和全量矩阵的计算量变成了很大的问题。

我们回想一下原始的CBOW和skip-gram，需要计算V个分类的softmax，更新V x N的矩阵。简单说明一下HS和NS：

- Hierachical Softmax：把V分类问题变成log(V)次的二分类。
- Negative sampling：计算总问题的一个子集

记住，这些都不是word2vec独有的优化训练trick。Hierachical softmax是计算softmax的一个有效方案，是通用的。而negative sampling则是更直截了当地只更新一部分权重矩阵W'的向量



### 3.1 Hierachical Softmax

在HS中，使用了hoffman树，把预测V个输出，变成了预测一组01编码，进行层次分类。树的结构大致是这样的：

将所有待预测的词放在叶子节点，即存在V个叶子节点，深度为$logV$, 其他节点的数量为V-1。（假设是完全二叉树，比如3层，则第一层1个，第二层2个，第三层$2^2$ 个，非叶子节点$3= 1+2=2^2 -1$个。）如图：

![img](https://pic4.zhimg.com/80/v2-04806895e59d0af21e791d2c59dfd1c7_hd.jpg)

在哈夫曼树中，每个叶节点是词表中的一个词，每个非叶子节点对应一个v'向量，树的深度为L(w)，整颗树有V-1个非叶子节点和V个叶节点。假设输入单词是w_i，目标单词是w_o，那么n(w, i)表示从根节点到叶节点w路径中的第i个节点，v'(w, i)表示n(w, i)所对应的v'向量。	需要注意的是v’是每个节点的向量，每个叶子节点共享这些路径上节点的向量。对于原始结构在计算最终结果使用softmax最大似然，变成了计算每个节点的sigmoid判断下一个节点属于左节点还是右节点。

$$
\begin{array}{c}{p(n, \text { left })=\sigma\left(\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)} \\ {p(n, \text { right })=1-\sigma\left(\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)=\sigma\left(-\mathbf{v}_{n}^{\prime T} \cdot \mathbf{h}\right)}\end{array}
$$

最终要最大化的概率为

$$
p\left(\omega=\omega_{O}\right)=\prod_{j=1}^{L(\omega)-1} \sigma\left([[n(\omega, j+1)=\operatorname{ch}(n(\omega, j))]] \cdot \mathbf{v}_{n(w, j)}^{\prime} T_{\mathbf{h}}\right)
$$

其中，隐层h和之前定义的一致（即CBOW是多个词的向量平均，skip-gram是对应的权重向量）。$[[x]]$是特定的符号函数：

$$[[x]]=\{\begin{array}{ll}{1} & {\text { if } x \text { is true }} \\ {-1,} & {\text { otherwise }}\end{array}$$

比如，对于目标单词w2, 

$$\begin{aligned} p\left(w_{2}=w_{O}\right) &=p\left(n\left(w_{2}, 1\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 2\right), \text { left }\right) \cdot p\left(n\left(w_{2}, 3\right), \text { right) }\right.\\ &=\sigma\left(\mathbf{v}_{n\left(w_{2}, 1\right)}^{\prime} \mathbf{h}\right) \cdot \sigma\left(\mathbf{v}_{n\left(w_{2}, 2\right)}^{\prime} \mathbf{h}\right) \cdot \sigma\left(-\mathbf{v}_{n\left(w_{2}, 3\right)}^{\prime} T_{\mathbf{h}}\right) \end{aligned}$$

且有：

$$
\sum_{i=1}^{V} p\left(\omega_{i}=\omega_{O}\right)=1
$$

对于这部分权重的推导，可以参考之前的方法，分别对h和$v_j$求偏导

损失函数定义为：

$$
E=-\log p\left(\omega=\omega_{O} | \omega_{I}\right)=-\sum_{j=1}^{L(\omega)-1} \log \sigma\left(\left[[[\cdot]] \mathbf{v}_{j}^{\prime T} \mathbf{h}\right)\right.
$$

求偏导得到

$$
\frac{\partial E}{\partial \mathbf{v}_{j}^{\prime}}=\frac{\partial E}{\partial \mathbf{v}_{j}^{\prime} \mathbf{h}} \cdot \frac{\partial \mathbf{v}^{\prime} \mathbf{h}}{\partial \mathbf{v}_{j}^{\prime}}=\left(\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-t_{j}\right) \cdot \mathbf{h}
$$

权重更新为: 

$$
\mathbf{v}_{j}^{\prime(n e w)}=\mathbf{v}_{j}^{\prime (old)}-\eta\left(\sigma\left(\mathbf{v}_{j}^{\prime T} \mathbf{h}\right)-t_{j}\right) \cdot \mathbf{h}, \text { for } j=1,2, \ldots, L(\omega)-1
$$

可以看到，原来计算V 个$e_j - t_j$， 变成了计算 $log N$个节点的$\sigma_j - t_j$，从而提升了效率。反向传播的推导也是类似的。



### 3.2 Negative Sampling

在每次迭代的过程中，最终需要输出的真实上下文单词（正样本）是确定需要更新的，与此同时，并不需要对全部负样本进行更新。因此，可以使用某些概率分布对负样本进行采样更新。

对于负样本的采样方式，可以根据词频进行随机抽样，我们倾向于选择词频比较大的负样本，比如“的”，这种词语其实是对我们的目标单词没有很大贡献的。Word2vec则在词频基础上取了0.75次幂，减小词频之间差异过大所带来的影响，使得词频比较小的负样本也有机会被采到。


$$
\text {weight}(w)=\frac{\operatorname{coun}(w)^{0.75}}{\sum_{u} \operatorname{count}(w)^{0.75}}
$$


在word2vec Parameter Learning Explained中，作者证明了使用下面简单的训练目标函数能够产生可靠的、高质量的 word embeddings。极大化正样本出现的概率，同时极小化负样本出现的概率，以sigmoid来代替softmax，相当于进行二分类，判断这个样本到底是不是正样本。


$$
E=-\log \sigma\left(\mathbf{v}_{\omega O}^{\prime T} \mathbf{h}\right) - \sum_{\omega_{j} \in W_{n e g}} \log \sigma\left(-\mathbf{v}_{\omega_{j}}^{\prime T} \mathbf{h}\right)
$$
这里不需要更新所有v'向量，只需要更新部分v'向量，这里的$w_j$是正样本$w_o$和负样本$w_{neg}$的集合，只更新这些样本所对应的v'向量。

## 4. 应用场景

实战有机会再写吧。