---
layout: post
title: IntelliJ IDEA 打包Jar
summary: 介绍如何使用IntelliJ IDEA 将自己的Java工程文件打包为单独一个可执行的jar，以及如何导入第三方jar包，并在导出自己的jar的同时捆绑第三方jar包。
featured-img: deep learning
language: chinese
category: others
---

## 导入第三方jar
- 从官网中下载相应的jar的压缩包，解压缩后将.jar文件拖拽进入工程文件夹下
- 在IDE中配置，将jar真正导入进来：右键project -> Open Module Settings -> Modules -> Dependencies -> Add(+) -> JARs or Directories -> 选择.jar文件存放的位置的jar -> 在导入进来这几个jar文件名字前打勾（千万不能忘记这一步！！！）-> Apply -> OK
- 当看到.java文件中import进来当jar不再有红色波浪线，说明导入成功

## 导出可执行文件jar
[参考文章](https://blog.csdn.net/englishfor/article/details/87631951)
- 右键project -> Open Module Settings -> Artifacts -> Add(+) -> JAR -> From modules with dependencies -> 选择main class -> 其他默认即可 -> OK
