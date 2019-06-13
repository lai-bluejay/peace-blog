---
<br>title: 用fastai解释什么是one-cycle-policy
<br>date: '2019-03-19 01:07:59'
<br>updated: '2019-04-14 22:54:43'
<br>tags:
<br>  - one-cycle
<br>  - fastai
<br>  - 学习率
<br>  - resnet
<br>categories:
<br>  - 深度学习
<br>abbrlink: e1f744e2
<br>---
<br><h2>fit one cycle</h2><br>
<br> 
<br>>"Manage 1-Cycle style training as outlined in Leslie Smith's [paper](https://arxiv.org/pdf/1803.09820.pdf)."
<br>
<br>  
<br>
<br><h2>what is one-cycle-policy?</h2><br>
<br>简单来说，one-cycle-policy, 使用的是一种周期性学习率，从较小的学习率开始学习，缓慢提高至较高的学习率，然后再慢慢下降，周而复始，每个周期的长度略微缩短，在训练的最后部分，学习率比之前的最小值降得更低。这不仅可以加速训练，还有助于防止模型落入损失平面的陡峭区域，使模型更倾向于寻找更平坦部分的极小值，从而缓解过拟合现象。
<br>
<br>  
<br>
<br><h3>one cycle 3步</h3><br>
<br>One-Cycle-Policy大概有三个步骤：
<br>
<br>  
<br>
<br>1. 我们逐渐将学习率从lr_max / div_factor提高到lr_max，同时我们逐渐减少从mom_max到mom_min的动量(momentum)。
<br>
<br>2. 反向再做一次：我们逐渐将学习率从lr_max降低到lr_max / div_factor，同时我们逐渐增加从mom_min到mom_max的动量。
<br>
<br>3. 我们进一步将学习率从lr_max / div_factor降低到lr_max /（div_factor x 100），我们保持动力稳定在mom_max。
<br>
<br>  
<br>
<br>我们来分别简单说明一下：
<br>
<br>  
<br>
<br>慢启动的想法并不新鲜：通常使用较低的值来预热训练，这正是one-cycle第一步实现的目标。Leslie不建议直接切换到更高的值，而是线性提高的最大值。
<br>
<br>  
<br>
<br>他在实验过程中观察到的是，在Cycle中期，高学习率将作为正则化的角色，并使NN不会过拟合。它们将阻止模型落在损失函数的陡峭区域，而是找到更平坦的最小值。他在[另一篇论文](https://arxiv.org/abs/1708.07120)中解释了通过使用这一政策，Hessian的近似值较低，表明SGD正在寻找更宽的平坦区域。
<br>
<br>  
<br>
<br>然后训练的最后一部分，学习率下降直到消失，将允许我们得到更平滑的部分内的局部最小值。在高学习率的同时，我们没有看到loss或acc的显着改善，并且验证集的loss有时非常高。但是当我们最终在最后降低学习率时，我们会发现这是很有好处的。
<br>
<br> 
<br><h3>reference</h3><br>
<br>具体的技术细节，可以参考Leslie 的论文。本文接下来的实验脚本都在[Cyclical LR and momentums.ipynb](https://github.com/lai-bluejay/Deep-Learning/blob/master/Cyclical%20LR%20and%20momentums.ipynb)
<br>
<br>  
<br>
<br>建议大家直接在colab运行这个脚本，在chrome中安装[open-in-colab extension](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo)，一键打开。
<br>
<br>  
<br>
<br>本文参考Sylvain Gugger的帖子[The 1cycle policy](https://sgugger.github.io/the-1cycle-policy.html)，稍微会有些不同。
<br>
<br>  
<br>
<br><h3>插播一句：Dropout 和 BN。</h3><br>
<br>Dropout 和BN在实际使用中，都会被提到加速训练，blabla...那会有什么缺点吗。
<br>
<br>一位神秘大佬有一天这样教育菜鸟：
<br>
<br>>**Dropout的缺点**
<br>dropout的那个是因为给BP回来的gradient加了一个很大的variance， 虽然sgd训练可以收敛， 但是由于方差太大，performance会变差。加了dropout之后，估计的gradient贼差，虽然还是无偏估计，但是variance就不可控了。
<br>对卷积层的正则化效果一般
<br>
<br>分割线
<br>>**BN的缺点**
<br>BN的是因为改变了loss，所以加了BN之后的网络跟原网络的最优解不是同一个，而且BN之后的网络的loss会跟batchsize强相关， 这就是不稳定的原因。
<br>其实BN的缺点在kaiming he的GN的那个paper里面已经说了，但是是从实验上说明的。
<br>只需要进一步的抽象一下，就知道了实验效果的差别是因为loss的差别。
<br>
<br>  
<br>
<br>身为PackageMan的我，赶紧掏出小本本记录。
<br>
<br>对于其中的方差偏移问题(variance shift)，具体可以参考：知乎专栏
<br>[大白话《Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift》](https://zhuanlan.zhihu.com/p/33101420)和机器之心[如何通过方差偏移理解批归一化与Dropout之间的冲突](https://www.jiqizhixin.com/articles/2018-01-23-4)
<br>
<br>  
<br>  
<br>
<br><h3>fastai中代码位置</h3><br>
<br>fastai/callbacks/one_cycle.py
<br>
<br>由于one_cycle只是用来帮助加快训练的，但是第一步还是需要找到一个初始的学习率，保证模型的loss在最初是一个较好的状态。我们通过实验来讲解代码。
<br>
<br>  
<br>
<br><h2>实验</h2><br>
<br>  
<br>
<br><h3>实验前准备</h3><br>
<br><h4>数据</h4><br>
<br>我们使用cifar10作为实验数据。
<br>
<br>```py
<br>
<br>from fastai.vision import *
<br><h2>下载数据并解压</h2><br>path = untar_data(URLs.CIFAR); path
<br>classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
<br>
<br>stats = (np.array([ 0.4914 , 0.48216, 0.44653]), np.array([ 0.24703, 0.24349, 0.26159]))
<br>
<br>size = 32
<br>batch_size = 512  # 如果性能不好，请调低到2^n，比如16，64
<br>
<br>def  get_data(bs):
<br>    ds_tfms = ([*rand_pad(4, 32), flip_lr(p=0.5)], [])
<br>    data = ImageDataBunch.from_folder(path, valid='test', classes=classes,ds_tfms=ds_tfms, bs=bs).normalize(cifar_stats)
<br>    return data
<br>
<br>data = get_data(batch_size)
<br>
<br>```
<br>
<br>  
<br>
<br><h4>ResNet</h4><br>
<br>在Sylvain的示例中，定义了ResNet和BasicBlock，但我这边建议，跟随最新的pytorch，避免在最新的fastai和pytorch使用中报错。
<br>
<br>  
<br>
<br>```python
<br>def conv3x3(in_planes, out_planes, stride=1):
<br>    """3x3 convolution with padding"""
<br>    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
<br>                     padding=1, bias=False)
<br>
<br>
<br>def conv1x1(in_planes, out_planes, stride=1):
<br>    """1x1 convolution"""
<br>    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
<br>  
<br>class BasicBlock(nn.Module):
<br>    expansion = 1
<br>
<br>    def __init__(self, inplanes, planes, stride=1, downsample=None):
<br>        super(BasicBlock, self).__init__()
<br>        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
<br>        self.conv1 = conv3x3(inplanes, planes, stride)
<br>        self.bn1 = nn.BatchNorm2d(planes)
<br>        self.relu = nn.ReLU(inplace=True)
<br>        self.conv2 = conv3x3(planes, planes)
<br>        self.bn2 = nn.BatchNorm2d(planes)
<br>        self.downsample = downsample
<br>        self.stride = stride
<br>
<br>    def forward(self, x):
<br>        identity = x
<br>
<br>        out = self.conv1(x)
<br>        out = self.bn1(out)
<br>        out = self.relu(out)
<br>
<br>        out = self.conv2(out)
<br>        out = self.bn2(out)
<br>
<br>        if self.downsample is not None:
<br>            identity = self.downsample(x)
<br>
<br>        out += identity
<br>        out = self.relu(out)
<br>
<br>        return out
<br>
<br>```
<br>
<br>  
<br>
<br>需要注意的是，在`fastai==1.0.48`中，构建cnn的函数直接变成了`cnn_learner`，同时要求`bash_arch`参数是Callable，因此我们需要一定的变化。具体参数查看`help(cnn_learner`.
<br>
<br>  
<br>
<br><h4>Have a look!</h4><br>
<br>参考fastai中resnet18的写法
<br>
<br>```python
<br>def resnet18(pretrained=False, **kwargs):
<br>    """Constructs a ResNet-18 model.
<br>    Args:
<br>        pretrained (bool): If True, returns a model pre-trained on ImageNet
<br>    """
<br>    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
<br>    if pretrained:
<br>        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
<br>    return model
<br>```
<br>  
<br>
<br>reference: [conv2d TypeError](https://forums.fast.ai/t/solved-error-while-creating-learner-for-senet-typeerror-conv2d-argument-input-position-1-must-be-tensor-not-bool/34203)
<br>
<br>  
<br>
<br>同时，我们不自己写ResNet，而是用fastai直接import的pytorch的定义，保证稳定~~
<br>
<br>```python
<br>
<br>def get_your_model_arch(pretrained=False, **kwargs):
<br>    model_arch = models.ResNet(BasicBlock, [9,9,9,1])
<br>    return model_arch
<br>model_arch = get_your_model_arch
<br><h2>model2 = models.resnet18()  get exception</h2><br>model2 = models.resnet18
<br>
<br>learn = cnn_learner(data, base_arch=model_arch, metrics=metrics.accuracy)
<br>learn.crit = F.nll_loss
<br>```
<br>
<br>
<br>到目前为止，一个CNN就构建好了，大家可以根据ResNet的参数，算一下这个是几层。
<br>
<br>  
<br>
<br><h3>lr_find(), 找到你的初始学习率</h3><br>
<br>定义好learner之后，如果要使用one-cycle-policy，那首先我们需要一个最佳的学习率。在fastai中，直接使用`learn.lr_find`就能找到。
<br>
<br>
<br>找到最佳学习率的思路参考：[How Do You Find A Good Learning Rate](https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html)
<br>
<br>  
<br>
<br>```python
<br>
<br>learn.lr_find(wd=1e-4,end_lr=100)
<br>
<br>```
<br>![findlr.png](https://img.hacpai.com/file/2019/03/findlr-19ca222d.png)
<br>
<br>我们从图中可以看到，学习率可以设置为0.08，初始的loss大概在2.6.
<br>
<br>  
<br>
<br><h3>实验</h3><br>
<br>One-Cycle-Policy大概有三个步骤：
<br>
<br>  
<br>
<br>1. 我们逐渐将学习率从lr_max / div_factor提高到lr_max，同时我们逐渐减少从mom_max到mom_min的动量(momentum)。
<br>
<br>2. 反向再做一次：我们逐渐将学习率从lr_max降低到lr_max / div_factor，同时我们逐渐增加从mom_min到mom_max的动量。
<br>
<br>3. 我们进一步将学习率从lr_max / div_factor降低到lr_max /（div_factor x 100），我们保持动力稳定在mom_max。
<br>
<br>  
<br>
<br>```python
<br>
<br>learn.recorder.plot_lr(show_moms=True)
<br>
<br>```
<br>
<br>大概得到如下图像：
<br>
<br>  
<br>![onecycleparams.png](https://img.hacpai.com/file/2019/03/onecycleparams-434586d5.png)
<br>  
<br>
<br><h4>实验1</h4><br>
<br>最大学习率: 0.08, 95轮one_cycle, weight decays:1e-4.
<br>
<br>在旧版本中，是`use_clr_beta`. 新版本中参数进行了修改。
<br>
<br>  
<br>
<br>- div_factor:10, pick 1/10th of the maximum learning rate for the minimum learning rate
<br>
<br>- pct_start: 0.1368, dedicate 13.68% of the cycle to the annealing at the end (that's 13 epochs over 95)
<br>
<br>- moms = (0.95, 0.85)
<br>
<br>- maximum momentum 0.95
<br>
<br>- minimum momentum 0.85
<br>
<br>  
<br>
<br>```python
<br>
<br>cyc_len = 95
<br>
<br>learn.fit_one_cycle(cyc_len=cyc_len, max_lr=0.08,div_factor=10, pct_start=0.1368,moms=(0.95, 0.85), wd=1e-4)
<br>
<br>```
<br>![exp195lrchanges.png](https://img.hacpai.com/file/2019/03/exp195lrchanges-42e6fed1.png)![exp195losses.png](https://img.hacpai.com/file/2019/03/exp195losses-8e008706.png)
<br>图中可以看到，学习率从0.08上升到0.8，最后下降到了很低的学习率。验证集的loss在中期变得不稳定，但在最后降了下来，同时，验证集和训练集的loss差并不大。
<br>
<br>  
<br>
<br><h4>实验2，更高的学习率//更小的cycle</h4><br>
<br>  
<br>  
<br>
<br>```python
<br>
<br><h2>修改cyc_len和div_factor, pct_start</h2><br>
<br>learn.fit_one_cycle(cyc_len=50, max_lr=0.8,div_factor=10, pct_start=0.5,moms=(0.95, 0.85), wd=1e-4)
<br>
<br>```
<br>
<br>同时，由于one-cycle学习率衰减的特性，允许我们使用更大的学习率。但为了保证loss不会太大，可能会需要更长的cycle来进行
<br>
<br>  
<br>![exp2biglr.png](https://img.hacpai.com/file/2019/03/exp2biglr-1bc6c635.png)
<br>学习率非常高，我们可以更快地学习并防止过度拟合。在我们消除学习率之前，验证损失和训练损失之间的差异仍然很小。这是Leslie Smith描述的超收敛现象(super convergence)。
<br>
<br>  
<br>
<br><h4>实验3 不变的动量momentum</h4><br>
<br>  
<br>
<br>```python
<br>
<br>learn.fit_one_cycle(cyc_len=cyc_len, max_lr=0.08,div_factor=10, pct_start=0.1368,moms=(0.9, 0.9), wd=1e-4)
<br>
<br>```
<br>
<br>  
<br>
<br>在Leslie的实验中，减少动量会带来更好的结果。由此推测，训练时，如果希望SGD快速下降到一个新的平坦区域，则新的梯度需要更多的权重。0.85~0.95是经验上的较好值。实验中，发现如果动量使用常数，在结果上，会和变化的动量在准确率上接近，但loss会更大一些。时间上，常数动量会快一些。
<br>
<br>  
<br>
<br><h2>其他参数的影响：weight decays</h2><br>
<br>![exp5wds.png](https://img.hacpai.com/file/2019/03/exp5wds-fd745f35.png)
<br>
<br>(横坐标是学习率，纵坐标是train_loss)
<br>
<br>  
<br>
<br>通过对wd的调整，发现不同的wd对学习率和loss是有很大影响的。这也是为什么在`lr_find(wd=1e-4)`，需要和`fit_one_cycle(wd=1e-4)`相同。
<br>
<br>  
<br>
<br><h2>结语</h2><br>
<br>one-cycle策略本身还是属于正则化的方法，也可以在模型训练中减少其他正则化方法的使用，因为one-cycle本来可能更有效，也允许在更大的初始学习率中训练更长时间。<br>