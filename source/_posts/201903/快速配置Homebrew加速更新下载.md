---
title: 快速配置Homebrew加速更新下载
date: '2019-03-10 12:40:59'
updated: '2019-03-23 12:03:00'
tags:
  - homebrew
  - 安利墙
  - 镜像地址
  - 镜像加速
categories:
  - '201903'
abbrlink: 4067f07b
---
快速配置Homebrew加速更新下载
--------------------

# 安装homebrew
首先确保你已经安装好了 Homebrew 了, 如果没有, 请参考 OPSX 指引页的 Homebrew 文档;

然后你只需要粘贴下述命令在对应终端运行.
```
/usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
```

使用 Homebrew 安装 Apple 没有预装但[你需要的东西](https://formulae.brew.sh/formula/ "Homebrew 软件包列表")。

Homebrew 会将软件包安装到独立目录，并将其文件软链接至`/usr/local`。
详情参考[Homebrew](https://brew.sh/index_zh-cn)

### Bash 终端配置

```
 # 替换brew.git:
    cd "$(brew --repo)"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/brew.git
    # 替换homebrew-core.git:
    cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/homebrew-core.git
    # 应用生效
    brew update
    # 替换homebrew-bottles:
    echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.bash_profile
    source ~/.bash_profile
```

### Zsh 终端配置

```
 # 替换brew.git:
    cd "$(brew --repo)"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/brew.git
    # 替换homebrew-core.git:
    cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
    git remote set-url origin https://mirrors.aliyun.com/homebrew/homebrew-core.git
    # 应用生效
    brew update
    # 替换homebrew-bottles:
    echo 'export HOMEBREW_BOTTLE_DOMAIN=https://mirrors.aliyun.com/homebrew/homebrew-bottles' >> ~/.zshrc
    source ~/.zshrc
```

### 恢复默认配置

出于某些场景, 可能需要回退到默认配置, 你可以通过下述方式回退到默认配置.

首先执行下述命令:

```
 # 重置brew.git:
	$ cd "$(brew --repo)"
	$ git remote set-url origin https://github.com/Homebrew/brew.git
	# 重置homebrew-core.git:
	$ cd "$(brew --repo)/Library/Taps/homebrew/homebrew-core"
	$ git remote set-url origin https://github.com/Homebrew/homebrew-core.git
```

然后删掉 HOMEBREW_BOTTLE_DOMAIN 环境变量,将你终端文件

` ~/.bash_profile`

或者

` ~/.zshrc`

中

`HOMEBREW_BOTTLE_DOMAIN`

行删掉, 并执行

` source ~/.bash_profile`

或者

` source ~/.zshrc`

更新标签