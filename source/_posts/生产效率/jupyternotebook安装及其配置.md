---
title: jupyter notebook 安装及其配置
date: '2019-05-05 14:52:16'
updated: '2019-05-05 14:55:01'
tags:
  - jupyter
  - anaconda
categories:
  - 生产效率
abbrlink: 300c3a74
---
### 预备

安装完整版本的anaconda


### 1.创建配置文件

在安装了python环境中执行命令

```shell
# 创建配置文件
jupyter notebook --generate-config
```


OS X 的配置文件位置:

```

~/.jupyter/jupyter_notebook_config.py

```

  

### 2.修改**_config.py

  

主要修改如下代码段：
```python
#ipython目录，改成想放的目录
c.NotebookApp.ipython_dir = 'C:/ipython'

#ipynb目录，自行修改
c.NotebookApp.notebook_dir = 'C:/ipython'

# The default URL to redirect to from `/`
# c.NotebookApp.default_url = '/tree'

# 端口监听：允许所有端口
c.NotebookApp.ip = '0.0.0.0'

# 其他配置
# c.NotebookApp.port = 8888

#密码和token
# The string should be of the form type:salt:hashed-password.
# c.NotebookApp.password = u''

# c.NotebookApp.token = ''

```

#### 启动命令
```shell
jupyter notebook --port=3001 --ip=0.0.0.0
```

建议加入supervisor中，或者在tmux中运行，保证后台运行

### 3.使用多核的jupyter notebook

首先，需要同时安装了2.X和3.X的ipython 和jupyter。或者简单一点，安装两个版本的anaconda。

```shell
#查看当前kernel列表
jupyter kernelspec list

#需要先切换到虚拟环境的bin目录`
python -m ipykernel install

```

可以用 --name指定参数

具体帮助文档：

```shell

$ python -m ipykernel install --help  
usage: ipython-kernel-install [-h] [--user] [--name NAME]  
                              [--display-name DISPLAY_NAME] [--prefix PREFIX]  
                              [--sys-prefix]  
Install the IPython kernel spec.  
optional arguments:optional arguments:  

  -h, --help            show this help message and exit

  --user                Install for the current user instead of system-wide

  --name NAME           Specify a name for the kernelspec. This is needed to

                        have multiple IPython kernels at the same time.

  --display-name DISPLAY_NAME

                        Specify the display name for the kernelspec. This is

                        helpful when you have multiple IPython kernels.

  --profile PROFILE     Specify an IPython profile to load. This can be used

                        to create custom versions of the kernel.

  --prefix PREFIX       Specify an install prefix for the kernelspec. This is

                        needed to install into a non-default location, such as

                        a conda/virtual-env.

  --sys-prefix          Install to Python's sys.prefix. Shorthand for

                        --prefix='/data/home/work/anaconda3/envs/vis_ana'. For

                        use in conda/virtual-envs.

```

  

#### Mac OS X或Unix-like 系统

```shell
#需要先切换到ipython所在目录（ipython2.x，ipython3.x），运行两次如下命令即可
./ipython kernel install

```

#### windows

需要在python 2 3的环境中分别运行。

```shell
ipython kernelspec install-self

```

#### 移除jupyter kernel

```shell
sudo ./ipython kernelspec uninstall kernel-name
```
### 4.自动重载import模块

```python

%load_ext autoreload

%autoreload 2

```  
  

### 5. 在服务器启动时不加载浏览器

启动时指定：  

>$ jupyter notebook --no-browser

或者 修改配置文件

c.NotebookApp.open_browser = False

  

### 6. 好用的插件

```shell
# jupyter lab
conda install -c conda-forge jupyterlab

# jupyter extension manager
conda install -c conda-forge jupyter_nbextensions_configurator

# jupyter dashboard
conda install -c conda-forge rise

# https://github.com/matplotlib/jupyter-matplotlib
conda install -c conda-forge ipympl

# If using the Notebook
conda install -c conda-forge widgetsnbextension

# If using JupyterLab
conda install nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
```  
