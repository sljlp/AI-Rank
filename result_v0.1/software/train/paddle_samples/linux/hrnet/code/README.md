<!-- omit in toc -->
# paddleseg HRNetW18 性能复现

此处给出了基于 [NGC PyTorch](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Segmentation) 实现的 HRNetW18 Pre-Training 任务的详细复现流程，包括执行环境、paddlepaddle版本、环境搭建、复现脚本、测试结果和测试日志。

<!-- omit in toc -->

## 目录
- [一、环境搭建](#一环境搭建)
  - [1. 单机8卡环境搭建](#1单机8卡环境搭建)
  - [2. python环境准备](#2python环境准备)
- [二、cityscapes数据集准备](#二cityscapes数据集准备)
- [三、测试步骤](#三测试步骤)
  - [1. 启动脚本](#1-启动脚本)
    - [1. 单卡启动](#1-单卡启动)
    - [2. 多卡启动](#2-多卡启动)
- [四、日志数据](#四日志数据)
- [五、测试结果](#五测试结果)


## 一、环境搭建  

### 1.单机8卡环境搭建
>4个节点环境一样    

我们遵循了 NGC PyTorch 官网提供的 [Quick Start Guide](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/LanguageModeling/BERT#quick-start-guide) 教程搭建了测试环境，主要过程如下：

- **拉取镜像**

```bash
    git clone https://github.com/NVIDIA/DeepLearningExamples
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    # 本次测试是在如下版本下完成的：
    git checkout 99b1c898cead5603c945721162270c2fe077b4a2
```

- **构建镜像**

```bash
    bash scripts/docker/build.sh   # 构建镜像
```

- **从[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)拉取模型代码**

```bash
    cd DeepLearningExamples/PyTorch/LanguageModeling/BERT
    git clone https://github.com/PaddlePaddle/PaddleSeg.git -b benckmark
    cd PaddleSeg
    # 本次测试是在如下版本下完成的：
    git checkout 36652c637a885165cf5be55f0c00e2cca3c629ce
```


- **启动镜像**
```bash
    bash scripts/docker/launch.sh  # 启动容器
```
    我们将 `launch.sh` 脚本中的 `docker` 命令换为了 `nvidia-docker` 启动的支持 GPU 的容器，同时将`BERT`(即`$pwd`)目录替换为`PaddleSeg`目录，其他均保持不变，脚本如下：
```bash
    #!/bin/bash

    CMD=${1:-/bin/bash}
    NV_VISIBLE_DEVICES=${2:-"all"}
    DOCKER_BRIDGE=${3:-"host"}

    nvidia-docker run --name test_bert_torch -it  \
    --net=$DOCKER_BRIDGE \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e LD_LIBRARY_PATH='/workspace/install/lib/' \
    -v $PWD/PaddleSeg:/workspace/PaddleSeg \
    -v $PWD/PaddleSeg/results:/results \
    PaddleSeg $CMD
```  

### 2.python环境准备  
python3.6 + [paddlepaddle](https://github.com/PaddlePaddle/Paddle) develop 版本(commit id: 84eca16dc158f0904e74f7abfcde05df81db8f02)

## 二、cityscapes数据集准备  

数据下载后的结构是open-mmlab官方显示的[文件树结构](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/dataset_prepare.md)，只使用了其中cityscapes的部分。  
>由于数据集比较大，且容易受网速的影响，为了更方便复现竞品的性能数据，我们使用了多进程下载方式，将原文件压缩包分为若干小包，下载之后再合并为整个包cityscapes.tar，然后进行解压

首次下载cityscapes数据时，可执行如下命令 
```bash
    # 下载cityscapes  
    wget https://paddleseg.bj.bcebos.com/dataset/cityscapes.tar  

    # 解压数据集
    tar -xzvf cityscapes.tar

    # checksum
    md5sum cityscapes.tar
    输出：cityscapes.tar 37724b19b6e5d41f9f147936d60b3c29

    # 放到 data/ 目录下
    mv cityscapes PaddleSeg/data/
```

## 三、测试步骤

为了更准确的测试 PaddleSeg-HRNetW18 在 `NVIDIA DGX-1 (8x V100 32GB)` 上的性能数据，我们严格按照官方提供的模型代码配置、启动脚本，进行了性能测试。

**重要的配置参数：**  
我们测试了GPU数目与AMP对精度和性能的影响，  
GPU数目分为`单机单卡`，`单机8卡`，`4机32卡`  
AMP分为是否开启AMP，开启AMP即`fp16`，不开启即`fp32`
 
代码提交到了[PaddleSeg](./PaddleSeg)中
### 1. 启动脚本  
在自己指定好训练所需配置之后，我们就可以进行接下来的启动训练了。
#### (1) 单卡启动  
 对于在单卡上的训练，我们的启动方式是根据[官方方式](https://github.com/PaddlePaddle/PaddleSeg/tree/benchmark#%E5%8A%A8%E6%80%81%E5%9B%BE)启动的。

```bash  
cd PaddleSeg
./run_single_card.sh > GPUx1-bs8.log 2>&1 &
```

#### (2) 多卡启动  

- **单机8卡**  
启动命令：
``` bash
cd PaddleSeg
./run_multi_card.sh 1 8 > GPUx8_time2train_ips-bs8.log 2>&1 &
```  

- **多机多卡**
``` bash
cd PaddleSeg
./run_multi_card.sh 4 8 > GPUx32_time2train_ips-bs4.log 2>&1 &
```

## 四、日志数据
>该模型训练是按照iter数训练的，每隔若干iter打印一次log，因此在打印AI-Rank-Log的时候，我们将iter数映射为epoch数目

- [单卡 samples_per_gpu=8](../log/GPUx1-bs8.log)
- [8卡 samples_per_gpu=8](../log/GPUx8_time2train_ips-bs8.log)
- [16卡 samples_per_gpu=2]
- [32卡 samples_per_gpu=4](../log/GPUx32_time2train_ips-bs4.log)

- [单卡 samples_per_gpu=4、AMP speed]
- [8卡 samples_per_gpu=4、AMP train]
- [16卡 samples_per_gpu=4、AMP]
- [32卡 samples_per_gpu=2、AMP train]


## 五、测试结果

#### fp32测试结果

|卡数 | Time2Train(sec) | 吞吐(samples/sec) |准确率(%) | 加速比|
|:-----:|:-----:|:-----:|:-----:|:-----:|
|1 | - | 17 | - | - |
|8 | 82322.66 | 96 | 79.3 | 5.65 |
|16| - |  | | |
|32| 22300.63 | 272  | 78.29 | 16 |

