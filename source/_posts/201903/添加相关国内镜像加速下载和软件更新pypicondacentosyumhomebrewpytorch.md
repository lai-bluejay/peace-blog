---
title: '添加相关国内镜像，加速下载和软件更新-pypi,conda,centos,yum,homebrew,pytorch'
date: '2019-03-10 10:54:31'
updated: '2019-03-23 12:01:31'
tags:
  - 镜像加速
  - mirrors
  - 镜像地址
  - 安利墙
categories:
  - '201903'
abbrlink: dd65a875
---
# 镜像参考地址
## 学校  按CTRL+F搜索
最好根据服务器地址选择最近的学校
*   中山大学镜像：[http://mirror.sysu.edu.cn/](http://mirror.sysu.edu.cn/)
*   山东理工大学：[http://mirrors.sdutlinux.org/](http://mirrors.sdutlinux.org/)
*   哈尔滨工业大学：[http://run.hit.edu.cn/](http://run.hit.edu.cn/)
*   中国地质大学：[http://cugbteam.org/](http://cugbteam.org/)
*   大连理工大学：[http://mirror.dlut.edu.cn/](http://mirror.dlut.edu.cn/)
*   西南林业大学[http://cs3.swfu.edu.cn/cs3guide.html](http://cs3.swfu.edu.cn/cs3guide.html)
*   北京化工大学（仅教育网可以访问），包含 CentOS 镜像：[http://ubuntu.buct.edu.cn/](http://ubuntu.buct.edu.cn/)
*   天津大学：[http://mirror.tju.edu.cn/](http://mirror.tju.edu.cn/)
*   西南大学：[http://linux.swu.edu.cn/swudownload/Distributions/](http://linux.swu.edu.cn/swudownload/Distributions/)
*   青岛大学：[http://mirror.qdu.edu.cn/](http://mirror.qdu.edu.cn/)
*   南京师范大学：[http://mirrors.njnu.edu.cn/](http://mirrors.njnu.edu.cn/)
*   大连东软信息学院：[http://mirrors.neusoft.edu.cn/](http://mirrors.neusoft.edu.cn/)
*   浙江大学：[http://mirrors.zju.edu.cn/](http://mirrors.zju.edu.cn/)
*   兰州大学：[http://mirror.lzu.edu.cn/](http://mirror.lzu.edu.cn/)
*   厦门大学：[http://mirrors.xmu.edu.cn/](http://mirrors.xmu.edu.cn/)
*   北京理工大学：  
    [http://mirror.bit.edu.cn](http://mirror.bit.edu.cn/)(IPv4 only)  
    [http://mirror.bit6.edu.cn](http://mirror.bit6.edu.cn/)(IPv6 only)
*   北京交通大学：  
    [http://mirror.bjtu.edu.cn](http://mirror.bjtu.edu.cn/)(IPv4 only)  
    [http://mirror6.bjtu.edu.cn](http://mirror6.bjtu.edu.cn/)(IPv6 only)  
    [http://debian.bjtu.edu.cn](http://debian.bjtu.edu.cn/)(IPv4+IPv6)
*   上海交通大学：  
    [http://ftp.sjtu.edu.cn/](http://ftp.sjtu.edu.cn/)(IPv4 only)  
    [http://ftp6.sjtu.edu.cn](http://ftp6.sjtu.edu.cn/)(IPv6 only)
*   清华大学：  
    [http://mirrors.tuna.tsinghua.edu.cn/](http://mirrors.tuna.tsinghua.edu.cn/)(IPv4+IPv6)  
    [http://mirrors.6.tuna.tsinghua.edu.cn/](http://mirrors.6.tuna.tsinghua.edu.cn/)(IPv6 only)  
    [http://mirrors.4.tuna.tsinghua.edu.cn/](http://mirrors.4.tuna.tsinghua.edu.cn/)(IPv4 only)
*   中国科学技术大学：  
    [http://mirrors.ustc.edu.cn/](http://mirrors.ustc.edu.cn/)(IPv4+IPv6)  
    [http://mirrors4.ustc.edu.cn/](http://mirrors4.ustc.edu.cn/)  
    [http://mirrors6.ustc.edu.cn/](http://mirrors6.ustc.edu.cn/)
*   东北大学：  
    [http://mirror.neu.edu.cn/](http://mirror.neu.edu.cn/)(IPv4 only)  
    [http://mirror.neu6.edu.cn/](http://mirror.neu6.edu.cn/)(IPv6 only)
*   华中科技大学：  
    [http://mirrors.hust.edu.cn/](http://mirrors.hust.edu.cn/)  
    [http://mirrors.hustunique.com/](http://mirrors.hustunique.com/)
*   电子科技大学：[http://ubuntu.uestc.edu.cn/](http://ubuntu.uestc.edu.cn/)  
    电子科大凝聚工作室(Raspbian单一系统镜像)[http://raspbian.cnssuestc.org/](http://raspbian.cnssuestc.org/)  
    电子科大星辰工作室(少数小众发布版镜像)[http://mirrors.stuhome.net/](http://mirrors.stuhome.net/)


## 云平台
*   阿里云开源镜像：[http://mirrors.aliyun.com/](http://mirrors.aliyun.com/)
*  腾讯云开源镜像：[https://mirrors.cloud.tencent.com/](https://mirrors.cloud.tencent.com/)

>若您使用腾讯云服务器，请将源的域名从 mirrors.cloud.tencent.com 改为 mirrors.tencentyun.com，使用内网流量不占用公网流量。


## 大公司
*   搜狐开源镜像站：[http://mirrors.sohu.com/](http://mirrors.sohu.com/)
*   网易开源镜像站：[http://mirrors.163.com/](http://mirrors.163.com/)
*   开源中国：[http://mirrors.oschina.net/](http://mirrors.oschina.net/)
*   首都在线科技股份有限公司：[http://mirrors.yun-idc.com/](http://mirrors.yun-idc.com/)
*   LUPA：[http://mirror.lupaworld.com/](http://mirror.lupaworld.com/)
*   常州贝特康姆软件技术有限公司(原cn99）：[http://centos.bitcomm.cn/](http://centos.bitcomm.cn/)


# 加速下载示例
以阿里云为例：

## centos yum加速

### 懒人版
如果是腾讯云直接覆盖git里的配置，里面配置都是服务器内网流量。
mysql使用中科大镜像。
```
git clone https://github.com/lai-bluejay/yum_repos_d.git
mv /etc/yum.repos.d/ /etc/yum.repos.d.bak/
mv yum_repos_d /etc/yum.repos.d
```

**1、备份**

`mv /etc/yum.repos.d/CentOS-Base.repo /etc/yum.repos.d/CentOS-Base.repo.backup`

**2、下载新的CentOS-Base.repo 到/etc/yum.repos.d/**
**注意自己的系统版本**
CentOS 7

`wget -O /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo`

或者

`curl -o /etc/yum.repos.d/CentOS-Base.repo http://mirrors.aliyun.com/repo/Centos-7.repo`

**更新缓存**
```
yum clean all
yum makecache
```

## python pip 加速下载
按照python和pip之后，创建配置文件夹
`mkdir ~/.pip`
之后，创建配置文件 `vim ~/.pip/pip.conf` :
```shell

[global]
index-url = https://mirrors.aliyun.com/pypi/simple/

[install]
trusted-host=mirrors.aliyun.com
```

## conda 清华源

conda也是很重要的一个python 数据科学使用的包和包管理器。

Anaconda 是一个用于科学计算的 Python 发行版，支持 Linux, Mac, Windows, 包含了众多流行的科学计算、数据分析的 Python 包。

Anaconda 安装包可以到[https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)下载。

TUNA 还提供了 Anaconda 仓库的镜像，运行以下命令:

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --set show_channel_urls yes 
```

即可添加 Anaconda Python 免费仓库。

运行`conda install numpy`测试一下吧。

### Miniconda 镜像使用帮助
Miniconda 是一个 Anaconda 的轻量级替代，默认只包含了 python 和 conda，但是可以通过 pip 和 conda 来安装所需要的包。

Miniconda 安装包可以到[https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)下载。

### pytorch
pytorch按照镜像源的说明，会报错，因为搜索路径已经修改。如下为正确配置：
[关于pytorch搜索路径的解释](https://github.com/tuna/issues/issues/112#issuecomment-457445594)

pytorch 的新版本安装，请指定channel。默认会搜[https://conda.anaconda.org/pytorch/linux-64/](https://conda.anaconda.org/pytorch/linux-64/)

```shell
conda install pytorch torchvision cuda80 -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/
```

## homebrew
参考另一篇文章
[快速配置Homebrew加速更新下载](https://jithub.cn/articles/2019/03/10/1552192859653.html)

## mysql下载
更新mysql的源如下。
[mysql-community.repo](https://github.com/lai-bluejay/yum_repos_d/blob/master/mysql-community.repo)
[mysql-community-source](https://github.com/lai-bluejay/yum_repos_d/blob/master/mysql-community-source.repo)