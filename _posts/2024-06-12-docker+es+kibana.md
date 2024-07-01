---
layout: post
title: docker的部署以及ES、Kibana基于docker的部署
summary: 在MAC中安装docker，并在docker中部署elasticsearch & kibana
featured-img: machine learning
language: chinese 
category: docker
---
# 写在最前面!!!

1. ElasticSearch的版本与Kibana的版本要保持一致
    - 建议先从官网中找到符合本机的Kibana镜像版本，再去下载与其版本一致的ElasticSearch镜像
2. 修改配置文件，可以不直接进入镜像包进行修改，因为一般镜像中并没安装vi/vim类似的命令
    - 复制到本地进行修改，修改好后再复制回去
```shell
# 复制到本地
docker cp es:/usr/share/elasticsearch/config/elasticsearch.yml ./
# 复制回去
docker cp ./elasticsearch.yml es:/usr/share/elasticsearch/config/elasticsearch.yml
```

# Mac中安装Docker 

- 主机：MAC Apple M1
- 登录/注册 [Docker官网](https://hub.docker.com/)，根据本地主机信息下载 Docker Desktop APP

# ElasticSearch基于Docker的部署

## 拉取指定版本的镜像

```shell
docker pull docker.elastic.co/elasticsearch/elasticsearch:8.6.2  
```


## 创建网络

```shell
docker network create es-net
# 删除网络 docker network rm es-net
# es-net 为自定义的名字，可作修改
## 若报错Error response from daemon: error while removing network: network es-net id XXX has active endpoints：
### 由于网络中还存在活跃的端点
# 查看活跃的端点  docker network inspect es-net
```

## 启动单个节点的es

```shell
docker run -d --name es \
-e "ES_JAVA_OPTS=-Xms512m -Xmx512m" \
-e "discovery.type=single-node" \
-v es-data:/usr/share/elasticsearch/data \
--privileged --network es-net \
-p 9202:9200 -p 9300:9300 docker.elastic.co/elasticsearch/elasticsearch:8.6.2 
```

## 修改配置

为了方便起见，使用无需账号密码登录，修改 `elasticsearch.yml`，在最后添加

```xml
xpack.security.enabled: false
```

## 测试是否启动成功
1. 通过 `docker ps` 进行查看

如果看到有`elasticsearch`容器内容，说明启动成功

2. 通过 `curl` 命令查看 elasticsearch 健康状态

```shell
curl -X GET "http://0.0.0.0:9202/_cat/health?v"
# 主要关注status：green-正常, yellow-, red-不正常
```

3. 通过访问网页 `http://0.0.0.0:9202/` 进行查看

返回以下信息说明成功
```html
{
  "name" : "fd086f3931ef",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "TJlXCRpIQhqKFMytnFsnvA",
  "version" : {
    "number" : "8.6.2",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "2d58d0f136141f03239816a4e360a8d17b6d8f29",
    "build_date" : "2023-02-13T09:35:20.314882762Z",
    "build_snapshot" : false,
    "lucene_version" : "9.4.2",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

如果无法打开，则可能是端口号冲突。

## 安装插件

1. 去官网找到对应版本的插件，下载压缩包到本地
2. 进入容器的`plugins`目录下，新建一个文件夹用于存储该插件内容
3. 将解压后的内容copy到`plugins`目录下新建的文件夹里面
4. 重启容器 `docker restart <容器名>`
5. 再次进入容器(使用root账户) ` docker exec -u root -it <容器名> /bin/bash`
6. `./bin/elasticsearch-plugin list` 查看是否安装成功


# Kibana基于Docker的部署
## 拉取指定版本的镜像

```shell
docker pull docker.elastic.co/kibana/kibana:8.6.2  
```
## 连接elasticsearch启动Kibana
```shell
docker run -d \                                                                   
--name kibana \
-e ELASTICSEARCH_HOSTS=http://es:9200 \
--network=es-net \
-p 5601:5601  \
docker.elastic.co/kibana/kibana:8.6.2
```

## 测试是否启动成功

1. 通过 `docker ps` 进行查看

如果有`kibana`的容器，则说明成功

2. 通过访问网页 `http://0.0.0.0:5601/` 进行查看

如果无法正常打开，显示 'kibana is not ready yet'。一般情况是由于配置错误，主要关注`ELASTICSEARCH_HOSTS`是否正确
