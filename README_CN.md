[English](README_EN.md) | 简体中文

![⚡️FastDeploy](https://user-images.githubusercontent.com/31974251/185771818-5d4423cd-c94c-4a49-9894-bc7a8d1c29d0.png)

</p>
             
<p align="center">
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"><img src="https://img.shields.io/github/v/release/PaddlePaddle/FastDeploy?color=ffa"></a>
    <a href=""><img src="https://img.shields.io/badge/python-3.7+-aff.svg"></a>
    <a href=""><img src="https://img.shields.io/badge/os-linux%2C%20win%2C%20mac-pink.svg"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/graphs/contributors"><img src="https://img.shields.io/github/contributors/PaddlePaddle/FastDeploy?color=9ea"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/commits"><img src="https://img.shields.io/github/commit-activity/m/PaddlePaddle/FastDeploy?color=3af"></a>
    <a href="https://pypi.org/project/FastDeploy-python/"><img src="https://img.shields.io/pypi/dm/FastDeploy-python?color=9cf"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/issues"><img src="https://img.shields.io/github/issues/PaddlePaddle/FastDeploy?color=9cc"></a>
    <a href="https://github.com/PaddlePaddle/FastDeploy/stargazers"><img src="https://img.shields.io/github/stars/PaddlePaddle/FastDeploy?color=ccf"></a>
</p>

<p align="center">
    <a href="docs/README_CN.md"> 使用文档 </a>
    |
    <a href="https://baidu-paddle.github.io/fastdeploy-api/"> API文档 </a>
    |
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"> 更新日志 </a>
</p>

**⚡️FastDeploy**是一款**全场景**、**易用灵活**、**极致高效**的AI推理部署工具。提供📦**开箱即用**的**云边端**部署体验, 支持超过 🔥150+ **Text**, **Vision**, **Speech**和**跨模态**模型，并实现🔚**端到端**的推理性能优化。包括图像分类、物体检测、图像分割、人脸检测、人脸识别、关键点检测、抠图、OCR、NLP、TTS等任务，满足开发者**多场景、多硬件、多平台**的产业部署需求。

|      [Image Classification](examples/vision/classification)                                       |  [Object Detection](examples/vision/detection)                                                                                             | [Semantic Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                     | [Potrait Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                                         |
|:---------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/200465949-da478e1b-21ce-43b8-9f3f-287460e786bd.png' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">                                                                                                                    | <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">                                                                                                                                 |
| [**Image Matting**](examples/vision/matting)                 |  [**Real-Time Matting**](examples/vision/matting)           | [**OCR**](examples/vision/ocr)                  |[**Face Alignment**](examples/vision/facealign)
| <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px"> |<img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"  >                                                                                                                              |<img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">  |
| [**Pose Estimation**](examples/vision/keypointdetection)                                                                                     | [**Behavior Recognition**](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                                        |  [**NLP**](examples/text)                                                                                                                                                                                                           |[**Speech**](examples/audio/pp-tts)  
| <img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif' height="126px" width="190px">  |  <p align="left">**input** ：早上好今天是2020<br>/10/29，最低温度是-3°C。<br><br> <p align="left">**output**: [<img src="https://user-images.githubusercontent.com/54695910/200161645-871e08da-5a31-4736-879c-a88bb171a676.png" width="170" style="max-width: 100%;">](https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/parakeet_espnet_fs2_pwg_demo/tn_g2p/parakeet/001.wav)</p>|


## 近期更新

- 🔥 [**【三日部署直播课回放】**](https://aistudio.baidu.com/aistudio/course/introduce/27800)
- **社区交流**
    - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1hhvpb279-iw2pNPwrDaMBQ5OQhO3Siw) and chat with other community members about ideas
    - **微信**：扫描二维码，填写问卷加入技术社区，与社区开发者探讨部署的痛点与方案

     <div align="center">
      <img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg" width = "120" height = "120" />
      </div>

- 🔥 **2022.11.23：Release FastDeploy [release v0.8.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.8)**
    -  **🖥️ 服务端部署：支持更多的模型，推理性能进一步提升**  
        -  新增 PIPNet、FaceLandmark1000 [人脸对齐模型](./examples/vision/facealign)的部署支持；
        -  新增[视频超分系列模型](./examples/vision/sr) PP-MSVSR、EDVR、BasicVSR 部署示例；
        -  升级[YOLOv7部署代码](https://github.com/PaddlePaddle/FastDeploy/pull/611)，支持 predict 及 batch_predict；
        -  新增 [UIE服务化部署](./examples/text/uie) 案例；
        -  [测试功能] 新增OpenVINO后端Device设置，支持集显/独立显卡的调用；
    -  **📲 移动端和端侧部署：支持更多模型**
        -  新增Android图像分类、目标检测、语义分割、OCR、人脸检测 APK工程及示例.
       
        |<font size=3>图像分类</font>|<font size=3>目标检测</font>|<font size=3>语义分割</font>|<font size=3>文字识别</font>|<font size=3>人脸检测</font>|  
        |:---:|:---:|:---:|:---:|:---:|  
        |<font size=2>[工程代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/classification)</font>|<font size=2>[工程代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/detection)</font>|<font size=2>[工程代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/segmentation)|<font size=2>[工程代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/ocr)</font>|<font size=2>[工程代码](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/facedet)</font>|  
        |<font size=2>[扫码或点击链接<br>安装试用](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-MobileNetV1.apk)</font>|<font size=2>[扫码或点击链接<br>安装试用](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-PicoDet.apk)</font>| <font size=2>[扫码或点击链接<br>安装试用](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-Portrait-HumanSegV2-Lite.apk)</font>| <font size=2> [扫码或点击链接<br>安装试用](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-PP-OCRv2.apk)</font>|<font size=2> [扫码或点击链接<br>安装试用](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-SCRFD.apk)</font>|     
        | <img src=https://user-images.githubusercontent.com/54695910/203604502-991972a8-5a9c-49cd-9e58-ed8e2a942b9b.png height="90" width="100"> |<img src=https://user-images.githubusercontent.com/54695910/203604475-724be708-27d6-4e56-9c2f-ae2eca24118c.png height="90" width="100">|<img src=https://user-images.githubusercontent.com/54695910/203604459-9a2915bc-91dc-460c-bff6-a0e2584d2eff.png height="90" width="100">|<img src=https://user-images.githubusercontent.com/54695910/203604453-6ce0118e-7b93-4044-8a92-56f2ab65c26a.png height="90" width="100">|<img src=https://user-images.githubusercontent.com/54695910/203604418-7c9703b5-1805-457e-966c-5a6625f212ff.png height="90" width="100">|

- [**more releases information**](./releases)

## 目录

* <details open> <summary><b>📖 文档教程（点击可收缩）</b></summary><div>

   - 安装文档
        - [预编译库下载安装](docs/cn/build_and_install/download_prebuilt_libraries.md)
        - [GPU部署环境编译安装](docs/cn/build_and_install/gpu.md)
        - [CPU部署环境编译安装](docs/cn/build_and_install/cpu.md)
        - [IPU部署环境编译安装](docs/cn/build_and_install/ipu.md)
        - [Jetson部署环境编译安装](docs/cn/build_and_install/jetson.md)
        - [Android平台部署环境编译安装](docs/cn/build_and_install/android.md)
   - 快速使用
        - [Python部署示例](docs/cn/quick_start/models/python.md)
        - [C++部署示例](docs/cn/quick_start/models/cpp.md)
        - [Runtime Python使用示例](docs/cn/quick_start/runtime/python.md)
        - [Runtime C++使用示例](docs/cn/quick_start/runtime/cpp.md)
   - API文档(进行中)
        - [Python API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/python/html/)
        - [C++ API文档](https://www.paddlepaddle.org.cn/fastdeploy-api-doc/cpp/html/)
   - 性能调优
        - [量化加速](docs/cn/quantize.md)
   - 常见问题
        - [1. 如何配置模型部署的推理后端](docs/cn/faq/how_to_change_backend.md)
        - [2. Windows上C++ SDK如何使用](docs/cn/faq/use_sdk_on_windows.md)
        - [3. Android上如何使用FastDeploy](docs/cn/faq/use_sdk_on_android.md)(进行中)
        - [4. TensorRT使用中的一些技巧](docs/cn/faq/tensorrt_tricks.md)
        - [5. 如何增加新的模型](docs/cn/faq/develop_a_new_model.md)(进行中)
   - 更多FastDeploy部署模块
        - [服务化部署](./serving)
        - [Benchmark测试](./benchmark)
</div></details>

* **🖥️ 服务器端部署**
    * [Python SDK快速开始](#fastdeploy-quick-start-python)  
    * [C++ SDK快速开始](#fastdeploy-quick-start-cpp)
    * [服务端模型支持列表](#fastdeploy-server-models)
* **📲 移动端和端侧部署**
    * [Paddle Lite NPU部署](#fastdeploy-edge-sdk-npu)
    * [端侧模型支持列表](#fastdeploy-edge-models)
* **🌐 Web和小程序部署**  
    * [Web端模型支持列表](#fastdeploy-web-models)
* [**社区交流**](#fastdeploy-community)
* [**Acknowledge**](#fastdeploy-acknowledge)  
* [**License**](#fastdeploy-license)

## 🖥️ 服务端部署

<div id="fastdeploy-quick-start-python"></div>

<details close> <summary><b>Python SDK快速开始（点开查看详情）</b></summary><div>

#### 快速安装

##### 前置依赖
- CUDA >= 11.2、cuDNN >= 8.0、Python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### 安装GPU版本

```bash
pip install numpy opencv-python fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```
##### [Conda安装(推荐)](docs/cn/build_and_install/download_prebuilt_libraries.md)
```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```
##### 安装CPU版本

```bash
pip install numpy opencv-python fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### Python 推理示例

* 准备模型和图片

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* 测试推理结果
```python
# GPU/TensorRT部署参考 examples/vision/detection/paddledetection/python
import cv2
import fastdeploy.vision as vision

model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")
im = cv2.imread("000000014439.jpg")
result = model.predict(im.copy())
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

</div></details>

<div id="fastdeploy-quick-start-cpp"></div>

<details>
<summary><b>C++ SDK快速开始（点开查看详情）</b></summary><div>


#### 安装

- 参考[C++预编译库下载](docs/cn/build_and_install/download_prebuilt_libraries.md)文档  

#### C++ 推理示例

* 准备模型和图片

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* 测试推理结果

```C++
// GPU/TensorRT部署参考 examples/vision/detection/paddledetection/cpp
#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  namespace vision = fastdeploy::vision;
  auto model = vision::detection::PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                          "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                          "ppyoloe_crn_l_300e_coco/infer_cfg.yml");
  auto im = cv::imread("000000014439.jpg");

  vision::DetectionResult res;
  model.Predict(&im, &res);

  auto vis_im = vision::Visualize::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
}
```
</div></details>

更多部署案例请参考[模型部署示例](examples) .

<div id="fastdeploy-server-models"></div>

### 服务端模型支持列表 🔥🔥🔥🔥🔥

符号说明: (1)  ✅: 已经支持; (2) ❔: 正在进行中; (3) N/A: 暂不支持; <br>

<details open><summary><b> 服务端模型支持列表（点击可收缩）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198619323-c9b1cbce-1c1c-4f92-9737-4805c7c0ff2f.png" />
</div>

| 任务场景                   | 模型                                                                                           | API                                                                                                                                       | Linux   | Linux      | Win     | Win        | Mac     | Mac     | Linux       | Linux         | Linux         | Linux   |
|:----------------------:|:--------------------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------:|:-------:|:----------:|:-------:|:----------:|:-------:|:-------:|:-----------:|:-------------:|:-------------:|:-------:|
| ---                    | ---                                                                                          | ---                                                                                                                                       | X86 CPU | NVIDIA GPU | X86 CPU | NVIDIA GPU | X86 CPU | Arm CPU | AArch64 CPU | NVIDIA Jetson | Graphcore IPU | Serving |
| Classification         | [PaddleClas/ResNet50](./examples/vision/classification/paddleclas)                           | [✅](./examples/vision/classification/paddleclas) | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [TorchVison/ResNet](examples/vision/classification/resnet)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Classification         | [ltralytics/YOLOv5Cls](examples/vision/classification/yolov5cls)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Classification         | [PaddleClas/PP-LCNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/PP-LCNetv2](./examples/vision/classification/paddleclas)                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/EfficientNet](./examples/vision/classification/paddleclas)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/GhostNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/MobileNetV1](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/MobileNetV2](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/MobileNetV3](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/ShuffleNetV2](./examples/vision/classification/paddleclas)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/SqueeezeNetV1.1](./examples/vision/classification/paddleclas)                    | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/Inceptionv3](./examples/vision/classification/paddleclas)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Classification         | [PaddleClas/PP-HGNet](./examples/vision/classification/paddleclas)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ✅                           | ❔       |
| Classification         | [PaddleClas/SwinTransformer](./examples/vision/classification/paddleclas)                    | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/PP-YOLOE](./examples/vision/detection/paddledetection)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/PicoDet](./examples/vision/detection/paddledetection)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/YOLOX](./examples/vision/detection/paddledetection)                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/YOLOv3](./examples/vision/detection/paddledetection)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/PP-YOLO](./examples/vision/detection/paddledetection)                       | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/PP-YOLOv2](./examples/vision/detection/paddledetection)                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/Faster-RCNN](./examples/vision/detection/paddledetection)                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [PaddleDetection/Mask-RCNN](./examples/vision/detection/paddledetection)                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [Megvii-BaseDetection/YOLOX](./examples/vision/detection/yolox)                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [WongKinYiu/YOLOv7](./examples/vision/detection/yolov7)                                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [WongKinYiu/YOLOv7end2end_trt](./examples/vision/detection/yolov7end2end_trt)                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [WongKinYiu/YOLOv7end2end_ort_](./examples/vision/detection/yolov7end2end_ort)               | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [meituan/YOLOv6](./examples/vision/detection/yolov6)                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [ultralytics/YOLOv5](./examples/vision/detection/yolov5)                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [WongKinYiu/YOLOR](./examples/vision/detection/yolor)                                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [WongKinYiu/ScaledYOLOv4](./examples/vision/detection/scaledyolov4)                          | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [ppogg/YOLOv5Lite](./examples/vision/detection/yolov5lite)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Detection              | [RangiLyu/NanoDetPlus](./examples/vision/detection/nanodet_plus)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| KeyPoint               | [PaddleDetection/TinyPose](./examples/vision/keypointdetection/tiny_pose)                    | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| KeyPoint               | [PaddleDetection/PicoDet + TinyPose](./examples/vision/keypointdetection/det_keypoint_unite) | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| HeadPose               | [omasaht/headpose](examples/vision/headpose)                                                 | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Tracking               | [PaddleDetection/PP-Tracking](examples/vision/tracking/pptracking)                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| OCR                    | [PaddleOCR/PP-OCRv2](./examples/vision/ocr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| OCR                    | [PaddleOCR/PP-OCRv3](./examples/vision/ocr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/PP-LiteSeg](./examples/vision/segmentation/paddleseg)                             | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegLite](./examples/vision/segmentation/paddleseg)                        | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/HRNet](./examples/vision/segmentation/paddleseg)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/PP-HumanSegServer](./examples/vision/segmentation/paddleseg)                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/Unet](./examples/vision/segmentation/paddleseg)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Segmentation           | [PaddleSeg/Deeplabv3](./examples/vision/segmentation/paddleseg)                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceDetection          | [biubug6/RetinaFace](./examples/vision/facedet/retinaface)                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceDetection          | [Linzaer/UltraFace](./examples/vision/facedet/ultraface)                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceDetection          | [deepcam-cn/YOLOv5Face](./examples/vision/facedet/yolov5face)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ✅       |
| FaceDetection          | [insightface/SCRFD](./examples/vision/facedet/scrfd)                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceAlign              | [Hsintao/PFLD](examples/vision/facealign/pfld)                                               | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceAlign              | [Single430FaceLandmark1000](./examples/vision/facealign/face_landmark_1000)                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceAlign              | [jhb86253817/PIPNet](./examples/vision/facealign)                                            | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceRecognition        | [insightface/ArcFace](./examples/vision/faceid/insightface)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceRecognition        | [insightface/CosFace](./examples/vision/faceid/insightface)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceRecognition        | [insightface/PartialFC](./examples/vision/faceid/insightface)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| FaceRecognition        | [insightface/VPL](./examples/vision/faceid/insightface)                                      | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Matting                | [ZHKKKe/MODNet](./examples/vision/matting/modnet)                                            | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Matting                | [PeterL1n/RobustVideoMatting]()                                                              | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Matting                | [PaddleSeg/PP-Matting](./examples/vision/matting/ppmatting)                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Matting                | [PaddleSeg/PP-HumanMatting](./examples/vision/matting/modnet)                                | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Matting                | [PaddleSeg/ModNet](./examples/vision/matting/modnet)                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Video Super-Resolution | [PaddleGAN/BasicVSR](./)                                                                     | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Video Super-Resolution | [PaddleGAN/EDVR](./examples/vision/sr/edvr)                                                  | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Video Super-Resolution | [PaddleGAN/PP-MSVSR](./examples/vision/sr/ppmsvsr)                                           | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| Information Extraction | [PaddleNLP/UIE](./examples/text/uie)                                                         | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ✅                           | ❔                           | ❔       |
| NLP                    | [PaddleNLP/ERNIE-3.0](./examples/text/ernie-3.0)                                             | ❔                                                | ❔                        | ❔                        | ❔                        | ❔                       | ❔                     | ❔                          | ❔                           | ❔                           | ✅       |
| Speech                 | [PaddleSpeech/PP-TTS](./examples/text/uie)                                                   | ✅                                                | ✅                        | ✅                        | ✅                        | ✅                       | ✅                     | ✅                          | ❔                           | --                          | ✅       |

</div></details>

<div id="fastdeploy-edge-doc"></div>

## 📲 移动端和端侧部署 🔥🔥🔥🔥

<div id="fastdeploy-edge-models"></div>

### 端侧模型支持列表
  
<details open><summary><b> 端侧模型支持列表（点击可收缩）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198619323-c9b1cbce-1c1c-4f92-9737-4805c7c0ff2f.png" />
</div>

| 任务场景               | 模型                                                                                        | 大小(MB)   | Linux   | Android | iOS     | Linux                      | Linux                                | Linux                             | Linux                    | 更新中...  |
|:------------------:|:-----------------------------------------------------------------------------------------:|:--------:|:-------:|:-------:|:-------: |:------------------:|:------------------------------------:|:---------------------------------:|:------------------------:|:-------:|
| ---                | ---                                                                                       | ---      | ARM CPU | ARM CPU | ARM CPU | 瑞芯微NPU<br>RK3568/RK3588 | 瑞芯微NPU<br>RV1109/RV1126/RK1808 | 晶晨NPU <br>A311D/S905D/C308X | 恩智浦NPU<br>i.MX&nbsp;8M&nbsp;Plus | 更新中...｜ |
| Classification     | [PaddleClas/PP-LCNet](examples/vision/classification/paddleclas)                          | 11.9     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/PP-LCNetv2](examples/vision/classification/paddleclas)                        | 26.6     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/EfficientNet](examples/vision/classification/paddleclas)                      | 31.4     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/GhostNet](examples/vision/classification/paddleclas)                          | 20.8     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV1](examples/vision/classification/paddleclas)                       | 17       | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV2](examples/vision/classification/paddleclas)                       | 14.2     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/MobileNetV3](examples/vision/classification/paddleclas)                       | 22       | ✅       | ✅       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| Classification     | [PaddleClas/ShuffleNetV2](examples/vision/classification/paddleclas)                      | 9.2      | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/SqueezeNetV1.1](examples/vision/classification/paddleclas)                    | 5        | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/Inceptionv3](examples/vision/classification/paddleclas)                       | 95.5     | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/PP-HGNet](examples/vision/classification/paddleclas)                          | 59       | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Classification     | [PaddleClas/SwinTransformer_224_win7](examples/vision/classification/paddleclas)          | 352.7    | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_coco](examples/vision/detection/paddledetection)        | 4.1      | ✅       | ✅       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_lcnet](examples/vision/detection/paddledetection)       | 4.9      | ✅       | ✅       | ❔       | ❔                          | ✅                                    | ✅                                 | ✅                        | --      |
| Detection          | [PaddleDetection/CenterNet](examples/vision/detection/paddledetection)                    | 4.8      | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/YOLOv3_MobileNetV3](examples/vision/detection/paddledetection)           | 94.6     | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_tiny_650e_coco](examples/vision/detection/paddledetection)       | 4.4      | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/SSD_MobileNetV1_300_120e_voc](examples/vision/detection/paddledetection) | 23.3     | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_ResNet50vd](examples/vision/detection/paddledetection)           | 188.5    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLOv2_ResNet50vd](examples/vision/detection/paddledetection)         | 218.7    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | [PaddleDetection/PP-YOLO_crn_l_300e_coco](examples/vision/detection/paddledetection)      | 209.1    | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Detection          | YOLOv5s                                                                                   | 29.3     | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Face Detection     | BlazeFace                                                                                 | 1.5      | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Face Detection     | RetinaFace                                                                                | 1.7      | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| Keypoint Detection | [PaddleDetection/PP-TinyPose](examples/vision/keypointdetection/tiny_pose)                | 5.5      | ✅       | ❔       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| Segmentation       | [PaddleSeg/PP-LiteSeg(STDC1)](examples/vision/segmentation/paddleseg)                     | 32.2     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg-Lite](examples/vision/segmentation/paddleseg)                      | 0.556    | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/HRNet-w18](examples/vision/segmentation/paddleseg)                             | 38.7     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg](examples/vision/segmentation/paddleseg)                           | 107.2    | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/Unet](examples/vision/segmentation/paddleseg)                                  | 53.7     | ✅       | ❔       | ❔       | ✅                          | --                                   | --                                | --                       | --      |
| Segmentation       | [PaddleSeg/Deeplabv3](examples/vision/segmentation/paddleseg)                             | 150      | ❔       | ❔       | ❔       | ✅                          |                                      |                                   |                          |         |
| OCR                | PaddleOCR/PP-OCRv1                                                                        | 2.3+4.4  | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| OCR                | [PaddleOCR/PP-OCRv2](examples/vision/ocr/PP-OCRv2)                                        | 2.3+4.4  | ✅       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |
| OCR                | [PaddleOCR/PP-OCRv3](examples/vision/ocr/PP-OCRv3)                                        | 2.4+10.6 | ✅       | ❔       | ❔       | ❔                          | ❔                                    | ❔                                 | ❔                        | --      |
| OCR                | PaddleOCR/PP-OCRv3-tiny                                                                   | 2.4+10.7 | ❔       | ❔       | ❔       | ❔                          | --                                   | --                                | --                       | --      |

</div></details>

## 🌐 Web和小程序部署

<div id="fastdeploy-web-models"></div>
 
<details open><summary><b> Web和小程序部署支持列表（点击可收缩）</b></summary><div>

| 任务场景               | 模型                                                                                          | [web_demo](examples/application/js/web_demo) |
|:------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ---                | ---                                                                                         | [Paddle.js](examples/application/js)         |
| Detection          | [FaceDetection](examples/application/js/web_demo/src/pages/cv/detection)                    | ✅                                            |
| Detection          | [ScrewDetection](examples/application/js/web_demo/src/pages/cv/detection)                   | ✅                                            |
| Segmentation       | [PaddleSeg/HumanSeg](./examples/application/js/web_demo/src/pages/cv/segmentation/HumanSeg) | ✅                                            |
| Object Recognition | [GestureRecognition](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| Object Recognition | [ItemIdentification](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| OCR                | [PaddleOCR/PP-OCRv3](./examples/application/js/web_demo/src/pages/cv/ocr)                   | ✅                                            |

</div></details>
  
<div id="fastdeploy-community"></div>

## 社区交流

- **加入社区👬：** 微信扫描二维码，进入**FastDeploy技术交流群**

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg"  width = "225" height = "225" />
</div>


<div id="fastdeploy-acknowledge"></div>

## Acknowledge

本项目中SDK生成和下载使用了[EasyEdge](https://ai.baidu.com/easyedge/app/openSource)中的免费开放能力，在此表示感谢。

## License

<div id="fastdeploy-license"></div>

FastDeploy遵循[Apache-2.0开源协议](./LICENSE)。
