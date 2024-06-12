---
layout: post
title: Spark
summary: 在MAC中安装docker，并在docker中部署elasticsearch & kibana
featured-img: machine learning
language: chinese 
category: docker
---

# RDD
弹性分布式数据集
## 弹性体现在哪里
- 自动进行内存和磁盘切换
- task如果失败会特定次数的重试
- stage如果失败会自动进行特定此时的重试，而且只会计算失败的分片

## RDD特性
- 高效的容错性
- 中间结果持久化保存到内存
- 存放的数据可以是Java对象，避免不必要的对象序列化和反序列化开销

## RDD之间的依赖关系
两者的区别**是否包含Shuffle操作**
### 窄依赖
- 不包括Shuffle操作
- 一个父RDD的分区只对应一个子RDD的一个分区
- 多个父RDD分区对应于一个子RDD分区
- e.g., 
### 宽依赖
- 包括Shuffle操作
- 一个父RDD的分区对应一个子RDD的多个分区
- e.g., 

## RDD 的操作
### 转换操作
- 返回的是RDD
- 不触发提交作业，只是完成作业中间过程处理，其操作具有惰性
- filter(fun)	筛选出满足函数fun的元素，并返回一个新的RDD
- map(fun)	对RDD中每个元素应用fun中，并将结果返回到一个新的RDD
- flatMap(fun)	将RDD中每个元素应用fun后，将其扁平化生成新的RDD
- reduceByKey(fun)	应用于RDD键值对，根据键Key，对其值Value进行fun操作，返回一个新的RDD键值对
- groupByKey()		应用于RDD键值对(K,V)，根据键Key，将其值Value组合成一个Iterable，返回一个新的RDD键值对(K,Iterable)   ️该操作没有参数
### 行动操作
- 返回的不是一个RDD
- 真正触发SparkContext提交作业
- reduce(fun)		通过函数fun聚合RDD中的元素，得到一个计算结果值
- fold(n)(fun)		通过函数fun将元素n与RDD中的元素一起聚合，得到一个计算结果
- foreach(fun)	对RDD中每个元素应用fun函数，得到结果值
- count()	返回RDD中的元素个数
- collect()	以数组的形式返回RDD的所有元素
- first()	返回RDD中的第一个元素
- take(n)		以数组形式返回RDD中前n个元素

## RDD分区
### 作用
- 增加并行度
- 减少通信开销

### 原则
尽量使分区的个数等于集群中CPU核心数目
- 在Local模式下默认为N
- standalone和YARN模式下默认为max{集群中所有CPU核心数总和，2}
- Mesos模式下默认为8
### 手动设置分区数量的方法
- 在创建RDD的时候就设置：
    - `sc.parallelize(data, Num)`
    - `sc.textFile(path, Num)`
- 使用repartition方法重新设置分区个数

## RDD运行过程及工作机制
- 创建RDD对象
- SparkContext负责计算RDD之间的依赖关系，构建DAG

> 什么是DAG：有向无环图，反映RDD之间的依赖关系

- DAGScheduler负责把DAG图分解为多个stage，每个stage中包含多个Task

> 划分stage的整体思路：从前往后，遇到宽依赖就断开，划分为一个stage；遇到窄依赖就把该RDD加入到该Stage中。

- 每个Task会被TaskScheduler分发给各个WorkerNode上的Executor去执行

## 什么是血缘关系
- RDD经过一系列转换操作，每一次都会产生不同的RDD，供给下一个转换操作使用，最后一个RDD经过行动操作，并输出到外部数据源。
- 记录的**粗颗粒度**的**转换**操作行为

## RDD的创建方式
- 通过textFile读取文件，从文件加载中生成
- 通过parallesize将List, Array转换，并通过并行集合创建

# Spark
## Spark运行架构
- 集群资源管理器Cluster Manager)
- 任务控制节点 (Driver)
    - 对应于一个Application
- 工作节点 (WorkerNode)
    - 对应于一个task
- 执行进程 (Executor)
    - 对应于一个WorkerNode
## Spark运行基本流程

客户端提交作业->Driver启动流程创建SparkContext->Driver申请资源并启动其余Executor-> Executor启动->作业调度（根据RDD的依赖关系构建DAG）生成Stages与Tasks->Task调度到Executor上，Executor启动线程执行Task->Driver管理Task状态->Task, Stage, Job完成

- 由任务控制节点(Driver)创建SparkContext对象，并由SparkContext与集群资源管理器Cluster Manager)进行通信
- 集群资源管理器Cluster Manager)为执行进程 (Executor)分配资源，并启动Executor进程，Executor运行情况将随着“心跳”发送到集群资源管理器Cluster Manager)上
- SparkContext创建RDD，并通过计算RDD之间的依赖关系，构建DAG
- 由DAG调度器（DAGScheduler）对DAG进行解析，进行stage划分，产生多个stages，每个stage是一个task集合
- 由任务调度器(TaskScheduler)将每个task分配到WorkNode上的Executor进行执行，同时，SparkContext将应用程序代码发放给Executor；
- task在Executor上运行，把执行结果反馈给任务调度器（TaskScheduler），然后反馈给DAG调度器（DAGScheduler），运行完毕后写入数据并释放所有资源


## 应用、作业、阶段、任务的关系

- 一个应用(Application)包含多个作业(Job)和一个任务控制节点(Driver)，其中阶段(stage)是作业(Job)的基本调度单位，一个作业(Job)包括多个阶段(stage)，每个阶段(stage)又包含多个任务(task)
- 应用(Application)：用户编写的Spark应用程序
- 作业(Job)：一个作业包括多个RDD以及作用于相应RDD上的各种操作
- 阶段(stage)：是作业的基本调度单位，一个作业会被分为多组任务，每组任务被称为一个阶段，代表一组关联但互相之间没有shuffle依赖关系（窄依赖）的任务组成的任务集
- 任务(task)：运行在Executor上的工作单元

![image-1](/assets/img/post_img/job.png)


## Spark的部署方式
- 单机模式(Local)
- 独立集群模式(Standalone)
- Spark on YARN
    - client: 用于交互式的作业
    - cluster: 用于企业生产环境
- Spark on Mesos

## Cache vs. Persist
- Cache：调用了Persist(MERORY_ONLY)方法，只是一个默认的缓存级别
- Persist可以根据情况设置其他的缓存级别

# Spark SQL
需要`SparkSession`对象
## DataFrame的创建
使用`spark.read`操作，从不同类型的文件中加载数据创建
- `spark.read.json('p.json')`
- `spark.read.csv('p.csv')`

## DataFrame的保存
使用`.write`操作，将数据保存成不同类型的文件
- `df.write.json('p.json')`
- `df.write.csv('p.csv')`

# Spark Streaming
需要`StreamContext`对象
## 基础输入源
### 文件流
需要Spark Streaming程序，一直对文件系统中的某个目录进行监听。
- `ssc.textFileStream()`

### 套接字流
通过Socket端口监听并接收数据
- `ssc.socketTextStream()`

### RDD队列流
- `ssc.queueStream()`

# Spark on Yarn-Cluster模式

1)Yarn-Cluster 第一步： Client向Yarn中提交应用程序，包括ApplicationMaster程序、启动ApplicationMaster的命令、需要在Executor中运行的程序等;

2)Yarn-Cluster 第二步：ResourceManager收到请求后，在集群中选择一个NodeManager，为该应用程序分配第一个Container，要求它在这个Container中启动应用程序的ApplicationMaster，其中ApplicationMaster进行SparkContext等的初始化;

3)Yarn-Cluster 第三步：ApplicationMaster向ResourceManager注册，这样用户可以直接通过ResourceManage查看应用程序的运行状态，然后它将采用轮询的方式通过RPC协议为各个任务申请资源，并监控它们的运行状态直到运行结束;

4)Yarn-Cluster 第四步：一旦ApplicationMaster申请到资源后，便与对应的NodeManager通信，要求它在获得的Container中启动启动Executor，启动后会向ApplicationMaster中的SparkContext注册并申请Task;

5)Yarn-Cluster 第五步：ApplicationMaster中的SparkContext分配Task给Executor执行，Executor运行Task并向ApplicationMaster汇报运行的状态和进度，以让ApplicationMaster随时掌握各个任务的运行状态，从而可以在任务失败时重新启动任务;

6)Yarn-Cluster 第六步：应用程序运行完成后，ApplicationMaster向ResourceManager申请注销并关闭自己。

# 数据倾斜
## 产生原因
在MapReduce过程中发生，由于key本身分布不均衡，或者shuffle时的并发度不够，使得过多的数据在同一个task中运行，把executor撑爆。
- 数据问题
1、key本身分布不均衡（包括大量的key为空）
2、key的设置不合理
- Spark使用问题
1、shuffle时的并发度不够
2、计算方式有误
## 解决方案
- 隔离执行：将异常的key（数据过多的key）过滤出来单独处理，最后与正常数据的处理结果进行union操作
- 对key先添加随机数，再对数据进行reduceByKey(func)，再把随机数去掉，再对数据进行一次reduceByKey(func)
- 使用reduceByKey代替groupByKey
- 调高shuffle的并行度

# Yarn中的关键组件
- ResourceManager(RM)
- NodeManager(NM)
- ApplicationMaster(AM)
- Container

# MapReduce vs. Spark
- MapReduce基于磁盘处理数据，Spark基于内存处理数据
    - MapReduce通过牺牲性能，减少了内存占用
    - Spark提高了处理数据的性能，被存到内存的数据可以反复使用
- Spark在处理数据的时候构建啦DAG图，从而减少了Shuffle和数据落地磁盘的次数，因此计算会比MapReduce快
- MapReduce是细粒度资源申请，Spark是粗粒度资源申请

> 粗粒度：在提交资源时，spark会提前向资源管理器（yarn，mess）将资源申请完毕，如果申请不到资源就等待，如果申请到就运行task任务，而不需要task再去申请资源。