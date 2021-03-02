# Triton指北

[TOC]

## 简介

[Triton](https://github.com/triton-inference-server/server)跟其他诸如TFServe、TorchServe一样，都是一个推理服务器，通过HTTP/REST、GRPC的网络协议的方式获取远程服务器的结果。能够挂载多种类型的模型，诸如：TensorRT、ONNX、PyTorch Script、TF的模型等。以及能够支撑CPU和GPU的共同访问。

## 适用场景

Triton的性能很好，能够应对绝大部分公司的前中期的高性能的模型推理。换句话说，比绝大部分公司的小伙伴自己基于Flask（或者其他server）+Pytorch（或者其他backend）等模式好很多，基本上能把GPU或者CPU跑满，再也不用为压榨服务器性能浪费宝贵时间了。

## 环境配置

### 布局

```bash
DB // 模型所在文件夹
├── 1 // 版本1文件夹
│   └── model.pt // 版本1中的模型
└── config.pbtxt // DB模型的配置
```

同一个模型可以有多个版本，可以通过client请求的时候来区分。

### 模型配置

triton的模型配置方便，通过配置文件即可。

创建配置文件config.pbtxt，下面展示一个简单的配置文件：

```text
name: "DB"
platform: "pytorch_libtorch"
max_batch_size: 4
input [
	{
		name: "INPUT__0"
		data_type: TYPE_FP32
		dims: [3,512,512]
	}
]
output [
	{
		name: "OUTPUT__0"
		data_type: TYPE_FP32
		dims: [1,512,512]
	}
]
```

解释如下：

- name：挂载的模型叫什么名字。最好英文，避免乱码。方便自己懂就行。
- platform：后端的平台。由于我们的模型就是pytorch的那一套，所以直接用这个即可。
- input：输入参数的配置。这里input可以理解是一个list(dict)，如果有多个输入，那么就跟上多个dict，逗号间隔。
  - name：由于pytorch的torchscript的特性，导致所有输入的名字无法像静态图一样自定义，这里的输入只能以`INPUT__下标`来命名。**其中INPUT后面是两个下划线。**
  - data_type：输入数据的类型。一般都是TYPE_FP32，如果做了其他骚操作，请自行修改。
  - dims：输入数据的维度，不需要跟上batch维度。
- output：与input类似。

更多详细配置，参见：[triton_model_configuration](https://github.com/triton-inference-server/server/blob/master/docs/model_configuration.md)

### server启动

建议直接上docker，避免环境安装的不对应。

`docker run --gpus=1 --name triton_inference_server -p8000:8000 -p8001:8001 -p8002:8002 -v/full/path/to/docs/examples/model_repository:/models nvcr.io/nvidia/tritonserver:<xx.yy>-py3 tritonserver --model-repository=/models`

解释：

- --gpus：配置triton能够使用的gpu的数量，可以写数字也可以全部添加`gpus=all`
- --name：配置当前container的名称
- -p：-p8000:8000 前面的8000为container内部端口，后面的8000为host的端口。如果没啥冲突，又想偷懒可以直接 --network=host直接让容器所有端口使用宿主机端口（**不推荐小白操作**）。
- -v：将host的磁盘的位置挂载在container中。前面的地址为host的地址，后面为对应container内部的地址。大家只需要修改前面的地址，后面的不用管。
- xx.yy：为triton的版本号，进入[triton release](https://github.com/triton-inference-server/server/releases)界面，可以用最新的，然后找到`NGC container`字样，例如`NGC container 21.02`其中`21.02`就是xx.yy。**后面客户端也必须和这里完全对应。**

> triton的镜像拉取很慢，大家如果网不好，可以自己build，自己参考一下triton的build的教程；当然也可以配置registry，拉取其他人上传到dockerhub的镜像。