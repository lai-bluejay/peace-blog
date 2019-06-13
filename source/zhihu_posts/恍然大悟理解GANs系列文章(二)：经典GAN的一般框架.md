
<br><h2>经典GAN的一般框架</h2><br>
<br>关于GAN的理论，可以把GAN模型按照正则化、非正则化模型分成两大类。非正则化包括经典GAN模型以及大部分变种，f-GAN就是关于经典GAN的一般框架的总结。这些模型的共同特点是对要生成的样本的分布不做任何先验假设，而是使用最小化差异的度量，尝试去解决一般性的数据样本生成问题。
<br>
<br>这篇内容不会特别重要，但是了解经典GAN的一般框架还是很有帮助的，同时，我们会看到由于不同的散度带来的问题（例如Mode collapse），为我们理解整个GAN的发展历程还是很有帮助的。。
<br>
<br><h3>1. f-divergence</h3><br>
<br>首先，我们了解一下什么是f-divergence。
<br>
<br>给定两个分布，P Q。$p(x)$ 和$q(x)$分别是x对应的概率，f-divergence定义如下：
<br><br><br><br>$D_f(P \| Q)=\int_{x} q(x) f\left(\frac{p(x)}{q(x)}\right) d x$<br><br><br><br>$\begin{aligned} D_{f}(P \| Q) &=\int_{x} q(x) f\left(\frac{p(x)}{q(x)}\right) d x \\ & \geq f\left(\int_{x} q(x) \frac{p(x)}{q(x)} d x\right) \\ & \geq f \left(\int_x p(x) dx\right) = 0\end{aligned}$<br>当：
<br>
<br>- $f(x) = xlog(x)$， 得到KL散度。
<br>
<br>- $f(x) = -log(x)$， 得到Reverse - KL散度。
<br>
<br>- $f(x) = (x-1)^2$，得到卡方散度。（Chi-square）
<br>
<br>    
<br>
<br>那，f-divergence和GANs有什么关系呢？这边涉及到一个Fenchel 共轭（Fenchel Conjugate）的问题。
<br>
<br><h3>2. Fenchel 共轭（Fenchel Conjugate）</h3><br>
<br>Fenchel共轭，是指，对于每个凸函数$f$，都有一个共轭函数$f^*$:
<br><br><br><br>$f^{*}(t)=\max _{x \in \operatorname{dom}(f)}\{x t-f(x)\}$<br>
<br>![image-20190612194225073](http://ww4.sinaimg.cn/large/006tNc79gy1g3ylyjh2tkj30xe0oyds5.jpg)
<br>
<br>​	
<br>
<br><h4>KL散度的验证 xlogx</h4><br>
<br>我们前面提到，KL散度的f-divergence定义为 $f(x) = xlogx$ ，此时
<br><br><br><br>$f^*(t)=\max _{x \in \operatorname{dom}(f)}\{x t-x \log x\}$<br><br><br><br>$f^{*}(t)=\exp (t-1) \times t-\exp (t-1) \times(t-1)=\exp (t-1)$<br>
<br>参考文档:[Nowozin, Sebastian, Botond Cseke, and Ryota Tomioka. "f-gan: Training generative neural samplers using variational divergence minimization." Advances in neural information processing systems. 2016.](https://arxiv.org/pdf/1606.00709.pdf)
<br>
<br><h3>3. 从共轭到f-GAN</h3><br>
<br>对于共轭函数，我们知道有（参考在SVM中提到的对偶问题）：
<br><br><br><br>$f(x)=\max _{t \in \operatorname{dom}\left(f^{*}\right)}\left\{x t-f^{*}(t)\right\}$<br><br><br><br>$\begin{aligned} D_{f}(P \| Q) &=\int_{x} q(x)\left(\max _{t \in \operatorname{dom}\left(f^{*}\right)}\left(\frac{p(x)}{q(x)} t-f^{*}(t)\right\}\right) d x \\ &\geq \int_{x} q(x)\left(\frac{p(x)}{q(x)} D(x)-f^{*}(D(x))\right) d x \\ & =\int_{x} p(x) D(x) d x-\int_{x} q(x) f^{*}(D(x)) d x \\   \end{aligned}$<br><br><br><br>$\begin{aligned} D_{f}(P \| Q) & \approx \max _{\mathrm{D}} \int p(x) D(x) d x-\int_{x} q(x) f^{*}(D(x)) d x \\ &=\max _{\mathrm{D}}\left\{E_{x \sim P}[D(x)]-E_{x \sim Q}\left[f^{*}(D(x))\right]\right\} \end{aligned}$<br><br><br><br>$\begin{aligned} G^{*} &=\arg \min _{G} D_{f}\left(P_{d a t a} \| P_{G}\right) \\ &=\arg \min _{G} \max _{D}\left\{E_{x \sim P_{\text {data}}}[D(x)]- E_{x \sim P_{G}}\left[f^{*}(D(x))\right] \}\right. \\ & =\arg \min _{G} \max _{D} V(G, D)  \end{aligned}$<br>
<br><h3>4. F-divergence List</h3><br>
<br>参考文档:[Nowozin, Sebastian, Botond Cseke, and Ryota Tomioka. "f-gan: Training generative neural samplers using variational divergence minimization." Advances in neural information processing systems. 2016.](https://arxiv.org/pdf/1606.00709.pdf)
<br>
<br>我们从文章中摘录一些f-divergence
<br>
<br>![image-20190613103807643](http://ww1.sinaimg.cn/large/006tNc79gy1g3zbuhbylbj30uo0boq6x.jpg)
<br>
<br>
<br>
<br>![image-20190613103929244](http://ww4.sinaimg.cn/large/006tNc79gy1g3zbvs4d06j30uu0fyafi.jpg)
<br>
<br>
<br>
<br><h3>5. GAN的其他问题</h3><br>
<br>上文我们说到，GAN的一般框架中，可以使用不同divergence进行生成器分布和实际分布差异的度量，我们来看看，生成的分布中间，和实际分布差别有多大。
<br>
<br>![image-20190613105144617](http://ww2.sinaimg.cn/large/006tNc79gy1g3zc8j4624j30tw13e7b7.jpg)
<br>
<br>在实验中，产生的分布的概率密度是双峰的，但实际生成的分布只有一个峰，或者接近右侧，或者处于中间（最大似然的生成函数往往处于中间）。就容易出现两个问题：
<br>
<br><h4>Mode Collapse/Mode Dropping</h4><br>
<br>生成的分布，就会集中在某个mode，就会出现：
<br>
<br>1. mode collapse，产生出来的图越来越接近；或者产生的图像和某张图像更接近
<br>2. mode dropping，产生的图像处于某个群。例如，生成人脸的时候，由于训练数据的分布，生成了很多白色肤色的人，但是没有生成黄色肤色的人。
<br>
<br>但是从上图我们也能发现，divergence的不同，并不能完全解决这个问题。
<br>
<br>我们总结一下经典GAN遇到的一些问题：
<br>
<br>1. 在实际过程中，生成数据和实际数据在低维流形上，重叠很少，意味着生成的数据分布和实际差别较大。（接近与两条曲线的交点，几乎可以忽略不计。
<br>
<br>2. 从另一个角度来看，在训练时，由于采用sampling的方式去度量divergence，因此数据分布间的重合变得很少。
<br>
<br>3. 以**JS散度**为例，当两个分布没有不接近的时候，JSD(P||Q) = log 2，无法度量直接倒地有多不接近。（相当于变成了0-1的问题）。这个问题就会导致Generator，在初始的位置会导致梯度为0或者很小，训练变得很难。上一篇文章中，我们说到把生成器换成$-log(D(\widetilde{x}))$。同时，另一个方案是退化Discriminator的能力，平衡生成器和判别器的训练程度。
<br>
<br>    
<br>
<br>接下来，我们就会将WGAN，
<br>
<br>- 彻底解决GAN训练不稳定的问题，不再需要小心平衡生成器和判别器的训练程度
<br>- 基本解决了collapse mode的问题，确保了生成样本的多样性 
<br>- 训练过程中终于有一个像交叉熵、准确率这样的数值来指示训练的进程，这个数值越小代表GAN训练得越好，代表生成器产生的图像质量越高
<br>- 以上一切好处不需要精心设计的网络架构，最简单的多层全连接网络就可以做到
<br>
<br>
<br>
<br>