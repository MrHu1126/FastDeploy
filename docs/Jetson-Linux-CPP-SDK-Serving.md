# 简介

本文档介绍FastDeploy中的模型SDK，在**Jetson Linux C++** 环境下：（1） **服务化**推理部署步骤，（2）介绍推理全流程API，方便开发者了解项目后二次开发。如果开发者对Jetson图像/视频部署感兴趣，可以参考[Jetson CPP Inference](./Jetson-Linux-CPP-SDK-Inference.md)文档。

**注意**：OCR目前不支持服务化推理部署。

<!--ts-->

* [简介](#简介)

* [环境准备](#环境准备)

* [快速开始](#快速开始)
  
  * [1. 项目结构说明](#1-项目结构说明)
  * [2. 测试 HTTP Demo](#2-测试-http-demo)
    * [2.1 启动HTTP预测服务](#21-启动http预测服务)

* [HTTP API介绍](#http-api介绍)
  
  * [1. 开启http服务](#1-开启http服务)
  * [2. 请求http服务](#2-请求http服务)
    * [2.1 http 请求方式一:不使用图片base64格式](#21-http-请求方式一不使用图片base64格式)
    * [2.2 http 请求方法二:使用图片base64格式](#22-http-请求方法二使用图片base64格式)
  * [3. http 返回数据](#3-http-返回数据)

* [FAQ](#faq)
  
  <!--te-->

# 环境准备

* Jetpack: 4.6 。安装Jetpack 4.6，参考[NVIDIA 官网-Jetpack4.6安装指南](https://developer.nvidia.com/jetpack-sdk-46)，或者参考采购的硬件厂商提供的安装方式进行安装。![]()
  
  | 序号  | 硬件               | Jetpack安装方式        | 下载链接                                                                                                                                        | ---- |
  | --- | ---------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------- | ---- |
  | 1   | Jetson Xavier NX | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jetson_xavier_nx/jetson-nx-jp46-sd-card-image.zip)      | ---- |
  | 2   | Jetson Nano      | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano/jetson-nano-jp46-sd-card-image.zip)         | ---- |
  | 3   | Jetson Nano 2GB  | SD Card Image      | [Download SD Card Image](https://developer.nvidia.com/embedded/l4t/r32_release_v6.1/jeston_nano_2gb/jetson-nano-2gb-jp46-sd-card-image.zip) | ---- |
  | 4   | agx xavier等      | NVIDIA SDK Manager | [Download NVIDIA SDK](https://developer.nvidia.com/nvsdk-manager)                                                                           | ---- |
  | 5   | 非官方版本，如emmc版     | 参考采购的硬件公司提供的安装指南   | ----                                                                                                                                        | ---- |
  
  注意：本项目SDK要求 `CUDA=10.2`、`cuDNN=8.2`、`TensorRT=8.0`、`gcc>=7.5` 、`cmake 在 3.0以上` ，安装 Jetpack4.6系统包后，CUDA、cuDNN、TensorRT、gcc和cmake版本就已经满足要求，无需在进行安装。

# 快速开始

## 1. 项目结构说明

根据开发者模型、部署芯片、操作系统需要，在图像界面[飞桨开源模型](https://ai.baidu.com/easyedge/app/openSource)或[GIthub](https://github.com/PaddlePaddle/FastDeploy)中选择对应的SDK进行下载。解压后SDK目录结构如下：

```
.EasyEdge-Linux-硬件芯片
├── RES  # 模型资源文件夹，一套模型适配不同硬件、OS和部署方式
│   ├── conf.json        # Android、iOS系统APP名字需要
│   ├── model            # 模型结构文件 
│   ├── params           # 模型参数文件
│   ├── label_list.txt   # 模型标签文件
│   ├── infer_cfg.json   # 模型前后处理等配置文件
├── ReadMe.txt
├── cpp                  # C++ SDK 文件结构
    └── baidu_easyedge_linux_cpp_x86_64_CPU.Generic_gcc5.4_v1.4.0_20220325.tar.gz
        ├── ReadMe.txt   
        ├── bin          # 可直接运行的二进制文件
        ├── include      # 二次开发用的头文件 
        ├── lib          # 二次开发用的所依赖的库
        ├── src          # 二次开发用的示例工程
        └── thirdparty   # 第三方依赖
```

## 2. 测试 HTTP Demo

> 模型资源文件（即压缩包中的RES文件夹）默认已经打包在开发者下载的SDK包中，请先将tar包整体拷贝到具体运行的设备中，再解压缩使用。

SDK中已经包含预先编译的二进制，可直接运行。以下运行示例均是`cd cpp/bin`路径下执行的结果。

### 2.1 启动HTTP预测服务

```
./easyedge_serving {模型RES文件夹路径} 1
```
**注意**：因为Jetson上http预测服务有缺陷，目前启动服务，需要多一个数字占位（譬如命令行中数字`1`）。开发者可以直接修改对应的源代码进行修改。

启动后，日志中会显示如下设备IP和24401端口号信息：

```
HTTP is now serving at 0.0.0.0:24401
```

此时，开发者可以打开浏览器，输入链接地址`http://0.0.0.0:24401`（这里的`设备IP和24401端口号`根据开发者电脑显示修改），选择图片来进行测试。

<div align=center><img src="https://user-images.githubusercontent.com/54695910/175855495-cd8d46ec-2492-4297-b3e4-2bda4cd6727c.png" width="600"></div>

同时，可以调用HTTP接口来访问服务，具体参考下文的[二次开发](#10)接口说明。

# HTTP API介绍

本章节主要结合[2.1 HTTP Demo]()的API介绍，方便开发者学习并将运行库嵌入到开发者的程序当中，更详细的API请参考`include/easyedge/easyedge*.h`文件。http服务包含服务端和客户端，目前支持的能力包括以下几种方式，Demo中提供了不使用图片base格式的`方式一：浏览器请求的方式`，其他几种方式开发者根据个人需要，选择开发。

## 1. 开启http服务

http服务的启动可直接使用`bin/easyedge_serving`，或参考`src/demo_serving.cpp`文件修改相关逻辑

```cpp
 /**
     * @brief 开启一个简单的demo http服务。
     * 该方法会block直到收到sigint/sigterm。
     * http服务里，图片的解码运行在cpu之上，可能会降低推理速度。
     * @tparam ConfigT
     * @param config
     * @param host
     * @param port
     * @param service_id service_id  user parameter, uri '/get/service_id' will respond this value with 'text/plain'
     * @param instance_num 实例数量，根据内存/显存/时延要求调整
     * @return
     */
    template<typename ConfigT>
    int start_http_server(
            const ConfigT &config,
            const std::string &host,
            int port,
            const std::string &service_id,
            int instance_num = 1);
```

## 2. 请求http服务

> 开发者可以打开浏览器，`http://{设备ip}:24401`，选择图片来进行测试。

### 2.1 http 请求方式一:不使用图片base64格式

URL中的get参数：

| 参数        | 说明        | 默认值              |
| --------- | --------- | ---------------- |
| threshold | 阈值过滤， 0~1 | 如不提供，则会使用模型的推荐阈值 |

HTTP POST Body即为图片的二进制内容(无需base64, 无需json)

Python请求示例

```Python
import requests

with open('./1.jpg', 'rb') as f:
    img = f.read()
    result = requests.post(
        'http://127.0.0.1:24401/',
        params={'threshold': 0.1},
        data=img).json()
```

### 2.2 http 请求方法二:使用图片base64格式

HTTP方法：POST
Header如下：

| 参数           | 值                |
| ------------ | ---------------- |
| Content-Type | application/json |

**Body请求填写**：

* 分类网络：
  body 中请求示例
  
  ```
  {
    "image": "<base64数据>"
    "top_num": 5
  }
  ```
  
  body中参数详情

| 参数      | 是否必选 | 类型     | 可选值范围 | 说明                                                                                  |
| ------- | ---- | ------ | ----- | ----------------------------------------------------------------------------------- |
| image   | 是    | string | -     | 图像数据，base64编码，要求base64图片编码后大小不超过4M,最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式 **注意去掉头部** |
| top_num | 否    | number | -     | 返回分类数量，不填该参数，则默认返回全部分类结果                                                            |

* 检测和分割网络：
  Body请求示例：
  
  ```
  {
    "image": "<base64数据>"
  }
  ```
  
  body中参数详情：

| 参数        | 是否必选 | 类型     | 可选值范围 | 说明                                                                                  |
| --------- | ---- | ------ | ----- | ----------------------------------------------------------------------------------- |
| image     | 是    | string | -     | 图像数据，base64编码，要求base64图片编码后大小不超过4M,最短边至少15px，最长边最大4096px，支持jpg/png/bmp格式 **注意去掉头部** |
| threshold | 否    | number | -     | 默认为推荐阈值，也可自行根据需要进行设置                                                                |

Python请求示例：

```Python
import base64
import requests
def main():
    with open("图像路径", 'rb') as f:
        result = requests.post("http://{服务ip地址}:24401/", json={
            "image": base64.b64encode(f.read()).decode("utf8")
        })
        # print(result.request.body)
        # print(result.request.headers)
        print(result.content)

if __name__ == '__main__':
    main()
```

## 3. http 返回数据

| 字段         | 类型说明   | 其他                                   |
| ---------- | ------ | ------------------------------------ |
| error_code | Number | 0为成功,非0参考message获得具体错误信息             |
| results    | Array  | 内容为具体的识别结果。其中字段的具体含义请参考`预测图像-返回格式`一节 |
| cost_ms    | Number | 预测耗时ms，不含网络交互时间                      |

返回示例

```json
{
    "cost_ms": 52,
    "error_code": 0,
    "results": [
        {
            "confidence": 0.94482421875,
            "index": 1,
            "label": "IronMan",
            "x1": 0.059185408055782318,
            "x2": 0.18795496225357056,
            "y1": 0.14762254059314728,
            "y2": 0.52510076761245728,
            "mask": "...",  // 图像分割模型字段
            "trackId": 0,  // 目标追踪模型字段
        },

      ]
}
```

*** 关于矩形坐标 ***

x1 * 图片宽度 = 检测框的左上角的横坐标

y1 * 图片高度 = 检测框的左上角的纵坐标

x2 * 图片宽度 = 检测框的右下角的横坐标

y2 * 图片高度 = 检测框的右下角的纵坐标

*** 关于分割模型 ***

其中，mask为分割模型的游程编码，解析方式可参考 [http demo](https://github.com/Baidu-AIP/EasyDL-Segmentation-Demo)

# FAQ

1. 如何处理一些 undefined reference / error while loading shared libraries?

> 如：./easyedge_demo: error while loading shared libraries: libeasyedge.so.1: cannot open shared object file: No such file or directory

遇到该问题时，请找到具体的库的位置，设置LD_LIBRARY_PATH；或者安装缺少的库。

> 示例一：libverify.so.1: cannot open shared object file: No such file or directory
> 链接找不到libveirfy.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../lib 解决(实际冒号后面添加的路径以libverify.so文件所在的路径为准)

> 示例二：libopencv_videoio.so.4.5: cannot open shared object file: No such file or directory
>  链接找不到libopencv_videoio.so文件，一般可通过 export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:../../thirdparty/opencv/lib 解决(实际冒号后面添加的路径以libopencv_videoio.so所在路径为准)

> 示例三：GLIBCXX_X.X.X  not found
> 链接无法找到glibc版本，请确保系统gcc版本>=SDK的gcc版本。升级gcc/glibc可以百度搜索相关文献。

2. 使用libcurl请求http服务时，速度明显变慢

这是因为libcurl请求continue导致server等待数据的问题，添加空的header即可

```bash
headers = curl_slist_append(headers, "Expect:");
```

3. 运行二进制时，提示 libverify.so cannot open shared object file

可能cmake没有正确设置rpath, 可以设置LD_LIBRARY_PATH为sdk的lib文件夹后，再运行：

```bash
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:../lib ./easyedge_demo
```

4. 编译时报错：file format not recognized

可能是因为在复制SDK时文件信息丢失。请将整个压缩包复制到目标设备中，再解压缩、编译。
