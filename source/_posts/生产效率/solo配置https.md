---
title: solo配置https
date: '2019-03-10 10:43:02'
updated: '2019-03-10 10:53:08'
tags:
  - linux
  - nginx
  - Solo
  - https
categories:
  - 生产效率
abbrlink: c0f4073
---
# solo的运行模式
请将`./WEB-INF/classes`路径下的配置做一个修改。
```
serverScheme=https
```
同时建议使用b3log的CDN静态资源加速
```
staticServerScheme=https
staticServerHost=static-solo.b3log.org
```

# 证书申请

各个云产品都可以申请证书，按照操作进行就好了。

证书审批通过后，将证书下载到服务器上。
可以参考[为iTerm2配置Zmodem文件传输(支持跳板机)](https://jithub.cn/articles/2019/03/09/1552136837606.html)

# 修改Nginx配置

```shell
upstream backend {
    server localhost:8080; # Tomcat/Jetty  原有的监听
}

server {
    listen       80;
    server_name  jithub.cn www.jithub.cn;

    access_log off;

        return 301 https://$server_name$request_uri;  # 监听80端口，并将server_name全部转发
}

server {
    listen       443;  # 修改监听接口
    server_name  jithub.cn www.jithub.cn;
    charset utf8; # 修改默认字符
    ssl on;  # 开启ssl

    # 很重要！！！设定你的ssl证书
    ssl_certificate /root/cert/Nginx/1_jithub.cn_bundle.crt;
    ssl_certificate_key /root/cert/Nginx/2_jithub.cn.key;

    # 重要！ 原有的接口代理可以不用修改，在内部使用http
    location / {
        proxy_pass http://backend$request_uri;
        proxy_set_header  Host $host:$server_port;
        proxy_set_header  X-Real-IP  $remote_addr;
        client_max_body_size  10m;
    }
}

```

# 重启Nginx
```service nginx restart
```