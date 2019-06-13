---
<br>title: 恍然大悟理解GANs系列文章(一)：GAN的直观理解和理论基础
<br>date: '2019-06-05 12:19:14'
<br>updated: '2019-06-05 12:19:14'
<br>tags:
<br>  - gan
<br>  - 深度学习
<br>  - 神经网络
<br>  - 生成式对抗网络
<br>mathjax: true
<br>categories:
<br>  - 深度学习
<br>abbrlink: f7632cc3
<br>---
<br>
<br><h3>brief of GAN GAN的简介</h3><br>
<br>>
<br>>
<br>>**这篇文章主要参考**
<br>>
<br>>1. 李宏毅老师的课程。[对抗生成网络(GAN)国语教程(2018)](https://www.bilibili.com/video/av24011528/?p=5)
<br>>2. Ian Goodfellow, NIPS 2016的slides[2016-12-04-NIPS](http://www.iangoodfellow.com/slides/2016-12-04-NIPS.pdf )
<br>>3. 沈鹏的翻译和整理 [GAN介绍 ](https://sinpycn.github.io/2017/04/27/GAN-Tutorial-Introduction.html)
<br>
<br>从分类上来说，GAN属于生成式模型。此处的生成式模型指的是用某种方式去学习并表达针对一个训练数据集分布的估计.
<br>
<br><h4>Generation 简述</h4><br>
<br>指vector -> G -> 图片，句子，文本，即提供一个随机向量，生成对应的object。对于NN来说，生成Generator，就是训练一个NN（set of function）。
<br>
<br>在GAN中，discriminator用于判断一个内容是不是真实存在的，值越大越真实。
<br>
<br><h4>why generative model</h4><br>
<br>同时也需要回答生成式模型的应用。
<br>
<br>1. 应用到强化学习中：
<br>    - 将产生式模型用于对假设环境的增强学习， 这样即使发生错误行为也不会造成实际的损失。 通过跟踪以前不同状态被访问的频率，或者不同的行为被尝试的频率，生成式模型也可以用于指导探索者/探险者。 生成式模型，特别是GAN，也可以用于inverse reinforcement learning。
<br>2. 可以使用有缺失的数据训练和预测
<br>3. 可以应对多任务学习
<br>4. 很多任务本身需要根据分布来产生真实数据。比如：低分辨率产生超高分辨率数据；艺术创作任务（随意画线产生图片）。
<br>
<br><h4>简单理解GAN的算法过程</h4><br>
<br>前面提到，生成式模型的基本流程都是：基于真实数据的分布$P_{data}$，训练一个生成器Generator，使得生成器产生的数据分布$P_G$与现实数据差异最小。在GAN当中，除了生成器之外，会有一个判别器（Discriminator），去判断生成器的好坏；生成器和判别器不断对抗训练，达到最终的目标；由于生成器和判别器的模型结构都是NN，所以得名生成式对抗网络(Generative Adversarial Networks, GAN)。
<br>
<br>我们先简单看一下GAN的算法过程，之后会详细讲述和其他生成式模型的区别，以及GAN的详细算法过程。
<br>
<br>
<br>
<br>```python
<br>初始化 G, D 的参数 
<br>for i in epochs:
<br>    抽样 m个x from real data;
<br>    抽样 m个z from a distribution（向量）;
<br>    （固定 G）：G(zi) = {x',..., x'} 
<br>    （update D）：max JSD(P_data||P_G); update theta-d (先教会老师，才能教学生，让G_1产生的vector获得低分)
<br>    (Fix D): 抽样 m个z from a distribution;
<br>    (Update G) : minimize: -log(D(G(zi))); update theta-g（让G产生的y_hat得到高分）
<br>如果两个模型具备足够的能力，那么最终博弈将达到纳什均衡，即G(z)符合实际分布，对所有x，D(x) = 1/2.
<br>```
<br>
<br>
<br>
<br>我们先直观理解GAN的目标函数：最大化目标函数，即最大化对真实数据的评分$D(x^{i})$, 最小化生成器产生的数据得分$D\left(\tilde{x}^{i}\right)$): 
<br><br><br><br>$\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(x^{i}\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(\tilde{x}^{i}\right)\right)$<br><br><br><br>$\theta_{d} \leftarrow \theta_{d}+\eta \nabla \tilde{V}\left(\theta_{d}\right)$<br><br><br><br>$\begin{array}{l}{ \tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log \left(1- D\left(G\left(z^{i}\right)\right)\right)} \\ { \theta_{g} \leftarrow \theta_{g} +\eta \nabla \tilde{V}\left(\theta_{g}\right)}\end{array}$<br>
<br>直观思考生成器和判别器：
<br>
<br>- 本质上，Generator的自我训练，也可以使用监督学习的方式。但是，可能并没有那么多的训练数据对。 
<br>
<br>- 而Discriminator也可以用于生成图片：在穷举所有可能的x，只需要得到最大的D(xi)，而xi便是生成的图片。但是求解argmax的计算量很难（线性方程还是可以穷举的）。
<br>
<br><br><br><br>$\widetilde{x}=\arg \max _{x \in X} D(x)$<br>
<br>
<br>
<br>
<br><h3>GAN ： structured learning</h3><br>
<br>在李宏毅老师的课程中提到，GAN属于结构化学习的一种。
<br>
<br>![image-20190604155856614](http://ww4.sinaimg.cn/large/006tNc79ly1g3p6je4z75j30wm0oo15i.jpg)
<br>
<br>结构化学习，即set of function，输出一个序列（翻译；asr）/矩阵（比如图片；text2img）/图/树等结构化的内容。和回归任务生成数值，分类任务生成类别不同。
<br>
<br>挑战在哪儿？
<br>
<br>- one-shot/ zero-shot learning：可以认为是极端的小样本学习
<br>- 要求算法有大菊观，能把各部件很好地组合
<br>
<br>GAN可以认为是解决Structured Learning的一种解决方案。传统的解决方案有两类：
<br>
<br>1. Bottom-up。相当于生成多个component，独立生成。（Generator）
<br>2. Top-Down。相当于从上帝视角评估整个方案，选择最好的组合，可以抓住多个component的相关关系。（Discriminator）
<br>
<br>同样，Auto-Encoder也属于类似的解决方案。
<br>
<br>补充学习：AE和VAE，保证不同的vector能够generate类似的结果。简单对比一下。
<br>
<br>- VAE对超参数不敏感，产生的数值会比较稳定
<br>- GANs对超参数敏感，产生数据的上下限变化较大
<br>- VAE使用MSE评估错误（即优化MSE），这个模型被强制为只能选择下一帧多个答案中的一个正确答案。 因为下一帧图像有很多的可能性， 这些可能的答案会有些细微的位置上的差别， 单一的答案选择让模型针对这些细微差别做了平均化处理，可能会使得生成的matrix糢糊。有很多任务，一个输入可能对应多个正确的输出， 每一个输出都是可接受的。 传统的基于平均值的机器学习模型， 比如， 对期望输出和预测输出的均方差（MSE）进行最小化的方法，无法训练此种有多个正确输出的模型。GAN能应对这种问题
<br>
<br>
<br>
<br><h3>Theory behind GAN</h3><br>
<br><h4>1. Maximum Likelihood  Estimation</h4><br>
<br>原有的生成式模型，使用最大似然的方式进行求解。由于要算最大似然，因此不能使用特别复杂的模型，所以一般预设高斯分布。
<br>
<br>给定一个实际分布$P_{data}(x)$, 同时假设有$P_G(x;\theta)$，和实际分布接近。（比如高斯混和分布））。
<br>
<br>
<br>
<br>过程如下：
<br>
<br>S1: 抽样m个样本：$\left\{x^1, x^2, \ldots, x^m\right\}$ ， 计算 $P_G(x^i;\theta)$， 即被G的分布抽样出来的概率。
<br>
<br>S2: 最大似然：$L=\prod\_{i=1}^m P\_G\left(x^i ; \theta\right)$，即最小似然的负对数。或者最小化KL散度（KL Divergence）（生成模型预设的分布与实际分布的KL散度）
<br>
<br>S3：更新$\theta$.
<br>
<br>$E\_{x \sim P\_{\text {data}}}$指从Pdata中抽样x。
<br>
<br>
<br><br><br><br>$\begin{aligned} \theta^{*} &=\arg \max _{\theta} \prod_{i=1}^{m} P_{G}\left(x^{i} ; \theta\right)=\arg \max _{\theta} \log \prod_{i=1}^{m} P_{G}\left(x^{i} ; \theta\right) \\ &=\arg \max _{\theta} \sum^{m} \log P_{G}\left(x^{i} ; \theta\right) \\ &\quad \approx \arg \max _{\theta} E_{x \sim P_{\text {data}}}\left[\log P_{G}(x ; \theta)\right] \\ &=\arg \max _{\theta} \int_{x} P_{\text {data}}(x) \log P_{G}(x ; \theta) d x-\int_{x} P_{\text {data}}(x) \log P_{\text {data}}(x) d x \\ &\quad=\arg \min _{\theta} K L\left(P_{\text {data}} \| P_{G}\right) \end{aligned}$<br>
<br>![Figure 9](https://sinpycn.github.io/images/201704/28/fig09.jpg)
<br>
<br>
<br>
<br>>引用翻译：
<br>>
<br>>深度生成模型可以使用最大似然的原理进行训练，不同方法的差异在于如何表达或者近似似然。 在此分类树的左边的分支， 模型构建了一个显式的密度函数，pmodel(x;θ)pmodel(x;θ)， 因此我们可以最大化一个显式的似然。 在这些使用显式密度的模型中， 密度函数的计算有的很容易，有的不容易， 这意味着，有时需要使用变分近似（variational approximations），或者Monte Carlo近似（或者两者）来最大化似然。 右边的分支，模型没有显式的表达一个数据空间的概率分布。 取而代之的是，模型提供一些方法来尽量少的直接的与概率分布打交道。 通常， 这种与概率分布交互的非直接方法提供了产生样本的能力。 有一些这种隐式的模型使用Markov chain来提供从数据分布中采样的能力，此类模型定义了一种通过随机转换一个已知的样本来得到属于同一个分布的另一个样本的方式。 有些方法可以在不需要任何输入情况下，通过一个简单的步骤产生一个样本。 尽管GAN使用的模型有时能被用来定义显式的密度， 但是GAN的训练算法仅仅使用模型的能力来产生样本。 因此， GAN使用最右边叶子的策略来训练的， 也就是使用一个隐式的模型通过此模型直接产生符合分布的样本。
<br>
<br>
<br>
<br>由于最大似然估计，一般使用正态分布或者高斯分布等进行数据抽样，一方面减少计算量，另一方面简化模型。那如何获得更一般的生成函数呢？
<br>
<br><h4>2. Generator</h4><br>
<br>对于更一般的生成器或者数据分布，我们想到了NN的通用近似定理，可以考虑使用NN，产生一个分布。
<br>
<br>1. 正态分布抽样（或者其他抽样）生成z
<br>2. 使用NN的生成器G，得到$x = G(z)$  ，通过神经网络，就得到了复杂的分布$P_G$。
<br>3. 目标：最小化$P\_G$ 和实际分布的差异$G^*=\arg \min\_G Div\left(P\_G, P\_{\text {data}}\right)$
<br>
<br>如果我们希望$P_G$完全的支持$x$空间，那么我们需要z的维数至少要和x一样大, 并且G是可微的， 以上为设计所需要的所有的条件。
<br>
<br>问题：在不知道分布的表达式的情况下，如何计算Div？
<br>
<br><h4>3. Discriminator</h4><br>
<br>第二小节讲的是生成器G的理想化形式，但最终我们也发现一个问题：在不知道分布的表达式(formula)的情况下，如何计算差异呢？
<br>
<br>退回到第一步和第二步，虽然我们不知道数据的实际分布是什么，但是，我们可以对数据进行抽样。抽样实际数据好理解，以图片生成为例：实际数据的抽样就是抽样出实际的图片；生成器G的抽样，就是随机抽样向量，通过G得到$G(z)$. 而，GAN中的Discriminator，就是度量两个分布差异的方案。
<br>
<br>这边只需要记住一句话：**训练一个判别器，就是最大化实际数据和生成器抽样数据的JS散度**。
<br><br><br><br>$G^{*}=\arg \min _{G} \max _{D} V(G, D)$<br>
<br>
<br>
<br><h4>3.1 Why  JSD （Jensen-Shannon Divergency)</h4><br>
<br>以下是简单的数学证明
<br>
<br>回顾一下之前GAN的判别器的目标函数:
<br><br><br><br>$\begin{aligned} V &=E_{x \sim P_{\text {data}}}[\log D(x)]+E_{x \sim P_{G}}[\log (1-D(x))] \\ &=\int_{x} P_{\text {data}}(x) \log D(x) d x+\int_{x} P_{G}(x) \log (1-D(x)) d x \\ &=\int_{x}\left[P_{\text {data}}(x) \log D(x)+P_{G}(x) \log (1-D(x))\right] d x \end{aligned}$<br>
<br>即，找到x，最大化$P\_{data}(x) \log D(x)+P\_G(x) \log (1-D(x))$。由于G和data都固定，可以认为是常数项，即最大化$\mathrm{f}(D)=a\operatorname{log}(D)+b \log (1-D)$。找极值点，即找微分=0的D。得到
<br>
<br><br><br><br>$D^{*}(x)=\frac{P_{d a t a}(x)}{P_{d a t a}(x)+P_{G}(x)}$<br>
<br>将$D^*$带入目标函数，并按照积分的形式展开，得到:
<br><br><br><br>$V(G, D) =E_{x \sim P_{\text {data}}}\left[\log \frac{P_{\text {data}}(x)}{P_{\text {data}}(x)+P_{G}(x)}\right] +E_{x \sim P_{G}}\left[\log \frac{P_{G}(x)}{P_{\text {data}}(x)+P_{G}(x)}\right] \\ =\int_{x} P_{d a t a}(x) \log \frac{P_{d a t a}(x)}{P_{d a t a}(x)+P_{G}(x)} d x +\int_{x} P_{G}(x) \log \frac{P_{G}(x)}{P_{\text {data}}(x)+P_{G}(x)} d x$<br><br><br><br>$V(G, D) =-2 \log 2+\int_{x} P_{d a t a}(x) \log \frac{P_{d a t a}(x)}{\left(P_{d a t a}(x)+P_{G}(x)\right) / 2} d x +\int_{x} P_{G}(x) \log \frac{P_{G}(x)}{\left(P_{d a t a}(x)+P_{G}(x)\right) / 2} d x \\ =-2 \log 2+\mathrm{KL}\left(\mathrm{P}_{\text { data }} \| \frac{\mathrm{P}_{\text { data }}+\mathrm{P}_{\mathrm{G}}}{2}\right)+\mathrm{KL}\left(\mathrm{P}_{\mathrm{G}} \| \frac{\mathrm{P}_{\text { data }}+\mathrm{P}_{\mathrm{G}}}{2}\right)$<br><br><br><br>$\operatorname{JSD}(P \| Q)=\frac{1}{2} D(P \| M)+\frac{1}{2} D(Q \| M) \\ M=\frac{1}{2}(P+Q)$<br><br><br><br>$V(G,D) =-2 \log 2+2 J S D\left(P_{\text {data}} \| P_{G}\right)$<br><h4>3.2 Why minimax??</h4><br>
<br>我们已经看到了目标函数是求实际样本和生成器产生样本的JS散度，但为什么是minimax的优化问题呢？有些地方会直接告诉你，GAN本来是生成器和判别器的零和博弈，即$J^{(G)}=-J^{(D)}$。而零和博弈被称为极小极大博弈。我们从更直观的角度来说明minimax在GAN中的意义。
<br>
<br>1. ** max what** :  在训练的第一步，我们需要提高判别器的辨别能力。即在给定生成器$G_i$的情况下，找到D的参数$\theta_d $最大化判别器对不同分布样本的JS散度（区分能力）。
<br>2. ** min what**： 与此同时，对于当前分辩能力最好的判别器$D_i$，找到$\theta_g$，使得判别器分辩两类样本的区分度最小。
<br>
<br>这就是所谓的内循环（判别器训练）的最大化，外循环（生成器训练）的最小化。理想状态下，对所有x，D(x) = 1/2，即随机划分。但是往往由于训练的种种不足，无法达到这个状态。
<br>
<br><h4>3.3 散度的选择</h4><br>
<br>KL散度不是对称的； 最小化$D\_{K L}\left(p\_{\text {model}}| | p\_{\text {data}}\right)$与最小化$D\_{K L}\left(p\_{d a t a} \| p\_{m o d e l}\right)$是不同的。
<br>
<br>![Figure 14](https://sinpycn.github.io/images/201705/03/fig14.jpg)
<br>
<br>
<br>
<br><h4>4. 更新过程</h4><br>
<br>在了解算法过程之前，我们来先看一个图。由于，一开始判别器是很容易判断假数据的，即$D(G(z))$会很小，那么根据log函数的图像，在算法初期就会下降很慢。因此，Ian Goodfellow把生成器的实际更新，变换成$-log(D(x))$，加快训练速度。 判别器最小化交叉熵， 但是生成器最大化同一个交叉熵。 这对生成器来说很不幸， 因为当判别器能够以很好的置信值成功拒绝生成样本时， 生成器的梯度将会消失。
<br>
<br>![image-20190604115547913](http://ww1.sinaimg.cn/large/006tNc79ly1g3ozif1a21j30be0j0whg.jpg)
<br>
<br>
<br>
<br>1. 初始化G  $\theta _g$，D  $\theta_d$
<br>2. 训练判别器，$\max_D V(G,D)$
<br>    1. 分布在$P_{data}$和$P_G$中，抽样m个数据。（$P_G$的抽样值，随机m个向量z，使得$\widetilde{x} = G(z)$），最大化目标函数
<br>
<br><br><br><br>$\tilde{V}=\frac{1}{m} \sum_{i=1}^{m} \log D\left(x^{i}\right)+\frac{1}{m} \sum_{i=1}^{m} \log \left(1-D\left(\tilde{x}^{i}\right)\right)$<br>​            固定G，训练D：$P_{data}$的数据为正样本，$P_G$为负样本，最小化交叉熵，训练“是否是生成样本”的二分类器
<br>
<br>​			  更新D的参数： 
<br><br><br><br>$\theta_{d} \leftarrow \theta_{d}+\eta \nabla \tilde{V}\left(\theta_{d}\right)$<br>3. 以上，变获得了JS散度的度量方式：判别器D。这边，可以训练多次D。
<br>
<br>4. 训练生成器，减少JS散度的绝对值大小$\min_G \max_D V(G, D)$
<br>
<br>    1. 抽样m个$P_G$.
<br>
<br>    2. 最小化目标函数
<br>        $$
<br>        \begin{array}{l}{ \tilde{V}=\frac{1}{m} \sum_{i=1}^{m}  - \log \left(D\left(G\left(z^{i}\right)\right)\right)} \end{array}
<br>        $$
<br>
<br>    3. 更新权重
<br>        $$
<br>       \theta_{g} \leftarrow \theta_{g} -\eta \nabla \tilde{V}\left(\theta_{g}\right)
<br>        $$
<br>
<br>    4. 注意：为了避免D和G的变换导致G的可行解范围发生很大变换，在训练过程中，应该让G的更新幅度要小。即更新一次就好。
<br>
<br>5. 重复K次 2~4，
<br>
<br><h3>4. GAN与其他生成模型的比较</h3><br>
<br>总而言之， GAN被设计为可以避免很多的其他生成模型的缺点：
<br>
<br>- 它可以并行产生样本， 而不是使用运行时间与xx的维数成比例的方法。 这一点比FVBN有优势。
<br>- 生成函数设计有很少的限制。 这一点是针对玻尔兹曼机的优势， 只有少数的概率分布能够被马尔可夫链来处理， 并且相比非线性ICA， 生成器必须是可逆的，并且隐变量编码z必须与样本x同维。
<br>- 不需要马尔可夫链。 这一点比玻尔兹曼机和生成随机网络有优势。
<br>- 不需要变分边界， 并且那些被用于GAN的模型是全局优化器， 所以GAN是渐近一致的（asymptotically consistent）。 一些VAE也可能是渐近一致，但是还没有被证明。
<br>- GAN通常被认为比其他方法可以产生更好的样本。
<br>
<br>同时， GAN也有一些缺点： 训练需要找到纳什均衡， 这是一个比优化一个目标函数更难的问题。<br>