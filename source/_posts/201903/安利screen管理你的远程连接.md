---
title: 安利screen-管理你的远程连接
date: '2019-03-09 12:27:43'
updated: '2019-03-10 13:45:15'
tags:
  - 安利墙
  - screen
  - linux技巧
categories:
  - '201903'
abbrlink: 74937c67
---
# 零. screen 是什么

> 引自[IBM的文章](https://www.ibm.com/developerworks/cn/linux/l-cn-screen/)：

你是不是经常需要 SSH 或者 telent 远程登录到 Linux 服务器？你是不是经常为一些长时间运行的任务而头疼，比如系统备份、ftp 传输等等。通常情况下我们都是为每一个这样的任务开一个远程终端窗口，因为他们执行的时间太长了。必须等待它执行完毕，在此期间可不能关掉窗口或者断开连接，否则这个任务就会被杀掉，一切半途而废了。  

  

screen 可以满足你的所有需求。

screen可以在多个进程对终端窗口进行多路复用，简单来说，就像是你**同时打开了多个窗口，并同时保持连接，可以随时关闭，随时恢复**，使用简单，只需要terminal即可完成所有操作。这样，就只需要打开一个terminal，就能拥有多个会话，并且不用担心后台执行。

  

# 一. 安装screen

```

yum install screen

apt-get install screen

```

甚至可以直接```pip install screen```。（好像可行。。。忘记了。。。）

  

# 二. 正确配置

默认配置文件在 ~/.screenrc中

启动 screen -c [config_file] -S [screen_name]

```shell  

# Set default encoding using utf8

defutf8on

## 解决中文乱码,这个要按需配置

defencodingutf8

encodingutf8utf8

#兼容shell 使得.bashrc .profile /etc/profile等里面的别名等设置生效

shell-$SHELL

#set the startup message

startup_messageoff

termlinux

## 解决无法滚动

termcapinfoxterm|xterms|xs ti@:te=\E[2J

# 屏幕缓冲区行数

defscrollback10000

# 下标签设置

hardstatuson

captionalways"%{= kw}%-w%{= kG}%{+b}[%n %t]%{-b}%{= kw}%+w %=%d %M %0c %{g}%H%{-}"

#关闭闪屏

vbelloff

#Keboard binding

# bind Alt+z to move to previous window

bindkey^[zprev

# bind Alt+x to move to next window

bindkey^[xnext

# bind Alt`~= to screen0~12

bindkey"^[`"select0

bindkey"^[1"select1

bindkey"^[2"select2

bindkey"^[3"select3

bindkey"^[4"select4

bindkey"^[5"select5

bindkey"^[6"select6

bindkey"^[7"select7

bindkey"^[8"select8

bindkey"^[9"select9

bindkey"^[0"select10

bindkey"^[-"select11

bindkey"^[="select12

# bind F5 to create a new screen

bindkey-k k5screen

# bind F6 to detach screen session (to background)

bindkey-k k6detach

# bind F7 to kill current screen window

bindkey-k k7kill

# bind F8 to rename current screen window

bindkey-k k8title

```

**# 三. 命令语法**

**```  
**# screen [-AmRvx -ls -wipe][-d <作业名称>][-h <行数>][-r <作业名称>][-s ][-S <作业名称>]  
**```**

**参数说明**

```shell
-A 　将所有的视窗都调整为目前终端机的大小。  
-d <作业名称> 　将指定的screen作业离线。  
-h <行数> 　指定视窗的缓冲区行数。  
-m 　即使目前已在作业中的screen作业，仍强制建立新的screen作业。  
-r <作业名称> 　恢复离线的screen作业。  
-R 　先试图恢复离线的作业。若找不到离线的作业，即建立新的screen作业。  
-s 　指定建立新视窗时，所要执行的shell。  
-S <作业名称> 　指定screen作业的名称。  
-v 　显示版本信息。  
-x 　恢复之前离线的screen作业。  
-ls或--list 　显示目前所有的screen作业。  
-wipe 　检查目前所有的screen作业，并删除已经无法使用的screen作业。

screen -X -S [screen_name]  kill  关闭一个screen
```

**# 四. 常用screen参数**
```shell
screen -S yourname -> 新建一个叫yourname的session  
screen -ls -> 列出当前所有的session  
screen -r yourname -> 回到yourname这个session  
screen -d yourname -> 远程detach某个session  
screen -d -r yourname -> 结束当前session并回到yourname这个session
```
**在每个screen session 下，所有命令都以 ctrl+a(C-a) 开始。**  
```shell
C-a ? -> 显示所有键绑定信息  
C-a c -> 创建一个新的运行shell的窗口并切换到该窗口  
C-a n -> Next，切换到下一个 window   
C-a p -> Previous，切换到前一个 window   
C-a 0..9 -> 切换到第 0..9 个 window  
Ctrl+a [Space] -> 由视窗0循序切换到视窗9  
C-a C-a -> 在两个最近使用的 window 间切换   
C-a x -> 锁住当前的 window，需用用户密码解锁  
C-a d -> detach，暂时离开当前session，将目前的 screen session (可能含有多个 windows) 丢到后台执行，并会回到还没进 screen 时的状态，此时在 screen session 里，每个 window 内运行的 process (无论是前台/后台)都在继续执行，即使 logout 也不影响。   
C-a z -> 把当前session放到后台执行，用 shell 的 fg 命令则可回去。  
C-a w -> 显示所有窗口列表  
C-a t -> Time，显示当前时间，和系统的 load   
C-a k -> kill window，强行关闭当前的 window  
C-a [ -> 进入 copy mode，在 copy mode 下可以回滚、搜索、复制就像用使用 vi 一样  
    C-b Backward，PageUp   
    C-f Forward，PageDown   
    H(大写) High，将光标移至左上角   
    L Low，将光标移至左下角   
    0 移到行首   
    $ 行末   
    w forward one word，以字为单位往前移   
    b backward one word，以字为单位往后移   
    Space 第一次按为标记区起点，第二次按为终点   
    Esc 结束 copy mode   
C-a ] -> Paste，把刚刚在 copy mode 选定的内容贴上
```