---
title: 为iTerm2配置Zmodem文件传输(支持跳板机)
date: '2019-03-09 21:07:17'
updated: '2019-03-10 13:46:47'
tags:
  - zmodem
  - iterm2
  - lrzsz
  - 安利墙
categories:
  - '201903'
abbrlink: 215bdd05
---
# 为iTerm配置Zmodem文件传输(支持跳板机)
## Zmodem
Zmodem是一种支持错误校验的文件传输协议，在它之前还有Xmodem、Ymodem。

其中包括两个命令（都是在服务器上运行）：
>sz：将文件发送到本地机器
>
>rz：从本地选择文件上传到服务器
>
>sz/rz 适合速度大约10k/s左右，适合传输小文件，还会弹出可视化窗口选择文件，很方便。

安装Zmodem
```shell
brew install lrzsz
```
下载脚本并赋予可执行权限：

```shell
wget https://raw.githubusercontent.com/mmastrac/iterm2-zmodem/master/iterm2-send-zmodem.sh -P /usr/local/bin/
wget https://raw.githubusercontent.com/mmastrac/iterm2-zmodem/master/iterm2-recv-zmodem.sh -P /usr/local/bin/
chmod +x /usr/local/bin/iterm2-send-zmodem.sh
chmod +x /usr/local/bin/iterm2-recv-zmodem.sh
```

## 配置iTerm
打开需要配置的Profile -> Advanced -> Triggers -> edit，按照下面格式添加两行：

```shell
Regular expression: rz waiting to receive.\*\*B0100 
Action: Run Silent Coprocess 
Parameters: /usr/local/bin/iterm2-send-zmodem.sh

  
Regular expression: \*\*B00000000000000 
Action: Run Silent Coprocess 
Parameters: /usr/local/bin/iterm2-recv-zmodem.sh
```
服务器也需要安装lrzsz。

## 如果Q&A
如果使用tmux或者screen等终端发起请求，请使用：

```
rz -e 
sz -e filename f2 f3 [...]
```
参考[zmodem](https://github.com/mmastrac/iterm2-zmodem)


## 跳板机
很多公司都使用跳板机登录开发环境，这种情况下我们需要使用zssh来登录跳板机，然后ssh到开发机后才能使用sz命令。

```shell
# 安装zssh：
brew install zssh

#登录跳板机：
zssh xxx@xxx

#登录开发机：
ssh xxx@zzz

#下载文件
sz filename # 发送filename
# 按Ctrl+@ 进入本地环境
pwd # 本地位置
rz # 接收
```