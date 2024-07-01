---
layout: post
title: Elasticsearch连接Nginx获取文件
summary: 本文介绍了Nginx基于Docker的部署，以及Elasticsearch连接Nginx获取文件
featured-img: machine learning
language: chinese 
category: docker
---

# 写在前面!!!

- 当`Elasticsearch`以及`Nginx`都部署在本地（即`0.0.0.0`），此时在`Elasticsearch`配置中连接`Nginx`，需要使用暴露出来的ip地址。而不能直接使用`0.0.0.0`。因为`0.0.0.0`默认是读取本服务自己的。

# 拉取Nginx镜像
```shell
docker pull nginx
```
# 创建挂载目录
```shell
mkdir /Users/handan/opt/nginx
mkdir /Users/handan/opt/nginx/conf
mkdir /Users/handan/opt/nginx/html
```
# 生成容器
```shell
docker run \
-p 80:80 \
--name nginx \
--network es-net \
-v /Users/handan/opt/nginx/conf/:/etc/nginx/conf/nginx.conf \
-v  /Users/handan/opt/nginx/html:/usr/share/nginx/html \
-d nginx
```

# 将容器文件拷贝到主机
```shell
docker cp nginx:/etc/nginx/nginx.conf /Users/handan/opt/nginx/conf/nginx.conf
docker cp nginx:/etc/nginx/conf.d /Users/handan/opt/nginx/conf/conf.d
docker cp nginx:/usr/share/nginx/html /Users/handan/opt/nginx/html
```

# 删除nginx容器 
```shell
docker rm -f nginx
```

# 重新运行镜像
```shell
docker run \
-p 80:80 \
--name nginx \
--network es-net \
-v /Users/handan/opt/nginx/conf/:/etc/nginx/conf/nginx.conf \
-v  /Users/handan/opt/nginx/html:/usr/share/nginx/html \
-d nginx
```

# `Elasticsearch`中读取在`nginx`中存放的自定义词典
## 在`nginx`的主目录下的`html`文件夹下创建文档

可以直接采用将宿主机（本地机器）编辑好的文件传到容器中
```bash
docker cp dictionary.txt nginx:/usr/share/nginx/html/dictionary.txt
```

## 重启`nginx`容器
```bash
docker restart nginx
```


## 修改`Elasticsearch`中的插件配置文件

修改`/usr/share/elasticsearch/plugins/ik/config/IKAnalyzer.cfg.xml`的`remote_ext_dict`
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
	<comment>IK Analyzer 扩展配置</comment>
	<!--用户可以在这里配置自己的扩展字典 -->
	<entry key="ext_dict"></entry>
	 <!--用户可以在这里配置自己的扩展停止词字典-->
	<entry key="ext_stopwords"></entry>
	<!--用户可以在这里配置远程扩展字典 -->
	<entry key="remote_ext_dict">http://<nginx的向外暴露ip>/dictionary.txt</entry>
	<!--用户可以在这里配置远程扩展停止词字典-->
	<!-- <entry key="remote_ext_stopwords">words_location</entry> -->
</properties>
```

## 重启`elasticsearch`容器
```bash
docker restart es
```
