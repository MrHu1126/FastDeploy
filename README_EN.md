English | [简体中文](README_CN.md)

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
    <a href="docs/README_EN.md"> Documents </a>
    |
    <a href="https://baidu-paddle.github.io/fastdeploy-api/"> API Docs </a>
    |
    <a href="https://github.com/PaddlePaddle/FastDeploy/releases"> Release Notes </a>
</p>

**⚡️FastDeploy** is an **Easy-to-use** and **High Performance** AI model deployment toolkit for Cloud, Mobile and Edge with 📦**out-of-the-box and unified experience**, 🔚**end-to-end optimization** for over **🔥150+ Text, Vision, Speech and Cross-modal AI models**.
Including image classification, object detection, image segmentation, face detection, face recognition, keypoint detection, matting, OCR, NLP, TTS and other tasks to meet developers' industrial deployment needs for **multi-scenario**, **multi-hardware** and **multi-platform**.

| [Image Classification](examples/vision/classification)                                                                                         | [Object Detection](examples/vision/detection)                                                                                                  | [Semantic Segmentation](examples/vision/segmentation/paddleseg)                                                                                  | [Potrait Segmentation](examples/vision/segmentation/paddleseg)                                                                                                                                                                                                                                                                                                                                                             |
|:----------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| <img src='https://user-images.githubusercontent.com/54695910/200465949-da478e1b-21ce-43b8-9f3f-287460e786bd.png' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054680-2f8d1952-c120-4b67-88fc-7d2d7d2378b4.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054711-6119f0e7-d741-43b1-b273-9493d103d49f.gif' height="126px" width="190px">   | <img src='https://user-images.githubusercontent.com/54695910/188054718-6395321c-8937-4fa0-881c-5b20deb92aaa.gif' height="126px" width="190px">                                                                                                                                                                                                                                                                             |
| [**Image Matting**](examples/vision/matting)                                                                                                   | [**Real-Time Matting**](examples/vision/matting)                                                                                               | [**OCR**](examples/vision/ocr)                                                                                                                   | [**Face Alignment**](examples/vision/facealign)                                                                                                                                                                                                                                                                                                                                                                            |
| <img src='https://user-images.githubusercontent.com/54695910/188058231-a5fe1ce1-0a38-460f-9582-e0b881514908.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054691-e4cb1a70-09fe-4691-bc62-5552d50bd853.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/188054669-a85996ba-f7f3-4646-ae1f-3b7e3e353e7d.gif' height="126px" width="190px"  > | <img src='https://user-images.githubusercontent.com/54695910/188059460-9845e717-c30a-4252-bd80-b7f6d4cf30cb.png' height="126px" width="190px">                                                                                                                                                                                                                                                                             |
| [**Pose Estimation**](examples/vision/keypointdetection)                                                                                       | [**Behavior Recognition**](https://github.com/PaddlePaddle/FastDeploy/issues/6)                                                                | [**NLP**](examples/text)                                                                                                                         | [**Speech**](examples/audio/pp-tts)                                                                                                                                                                                                                                                                                                                                                                                        |
| <img src='https://user-images.githubusercontent.com/54695910/188054671-394db8dd-537c-42b1-9d90-468d7ad1530e.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/48054808/173034825-623e4f78-22a5-4f14-9b83-dc47aa868478.gif' height="126px" width="190px"> | <img src='https://user-images.githubusercontent.com/54695910/200162475-f5d85d70-18fb-4930-8e7e-9ca065c1d618.gif' height="126px" width="190px">   | <p align="left">**input**:Life was like a box<br> of chocolates, you never<br> know what you're <br>gonna get.<br> <p align="left">**output**: [<img src="https://user-images.githubusercontent.com/54695910/200161645-871e08da-5a31-4736-879c-a88bb171a676.png" width="150" style="max-width: 100%;">](https://paddlespeech.bj.bcebos.com/Parakeet/docs/demos/tacotron2_ljspeech_waveflow_samples_0.2/sentence_1.wav)</p> |

## 📣 Recent Updates
- **Community**
  - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1hhvpb279-iw2pNPwrDaMBQ5OQhO3Siw) and chat with other community members about ideas.
  - **WeChat**：Scan the QR code below using WeChat, follow the PaddlePaddle official account and fill out the questionnaire to join the WeChat group.
    
    <div align="center">
    <img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg" width = "100" height = "100" />
    </div>

- 🔥 **2022.11.23：Release FastDeploy [release v0.8.0](https://github.com/PaddlePaddle/FastDeploy/tree/release/0.8.0)** <br>
  
  - **🖥️ Server-side and Cloud Deployment: Support more CV models, improve deployment performance**
    - Support [PIPNet](./examples/vision/facealign/pipnet), [FaceLandmark1000](./examples/vision/facealign/face_landmark_1000) face alignment models deployment;
    - Support [Video Super-Resolution](./examples/vision/sr) series model PP-MSVSR、EDVR、BasicVSR;
    - Upgrade YOLOv7 deployment code to add `batch_predict` deployment;
    - Support [UIE service-based](./examples/text/uie) deployment;
    - Add Python API to_dlpack interface for FDTensor to support copyless transfer of FDTensor between frameworks.
  - **📱 Mobile and Edge Device Deployment: support more CV model**
    - Support Android image classification, target detection, semantic segmentation, OCR, face detection APK projects and examples.
    
     |<font size=3>Image Classification</font>|<font size=3>Object Detection</font>|<font size=3>Semantic Segmentation</font>|<font size=3>OCR</font>|<font size=3>Face Detection</font>|  
    |:---:|:---:|:---:|:---:|:---:|  
    |<font size=2>[Project Code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/classification)</font>|<font size=2>[Project Code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/detection)</font>|<font size=2>[Project Code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/segmentation)|<font size=2>[Project Code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/ocr)</font>|<font size=2>[Project Code](https://github.com/PaddlePaddle/FastDeploy/tree/develop/java/android/app/src/main/java/com/baidu/paddle/fastdeploy/app/examples/facedet)</font>|  
    |<font size=2>[Scan&nbsp;the&nbsp;code<br>or&nbsp;click&nbsp;on&nbsp;the&nbsp;link<br>to&nbsp;install](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-MobileNetV1.apk)</font>|<font size=2>[Scan&nbsp;the&nbsp;code<br>or&nbsp;click&nbsp;on&nbsp;the&nbsp;link<br>to&nbsp;install](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-PicoDet.apk)</font>| <font size=2>[Scan&nbsp;the&nbsp;code<br>or&nbsp;click&nbsp;on&nbsp;the&nbsp;link<br>to&nbsp;install](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-Portrait-HumanSegV2-Lite.apk)</font>| <font size=2> [Scan&nbsp;the&nbsp;code<br>or&nbsp;click&nbsp;on&nbsp;the&nbsp;link<br>to&nbsp;install](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-PP-OCRv2.apk)</font>|<font size=2> [Scan&nbsp;the&nbsp;code<br>or&nbsp;click&nbsp;on&nbsp;the&nbsp;link<br>to&nbsp;install](https://bj.bcebos.com/fastdeploy/release/android/FastDeploy-SCRFD.apk)</font>|     
    | <img src=https://user-images.githubusercontent.com/54695910/203604502-991972a8-5a9c-49cd-9e58-ed8e2a942b9b.png height="100" width="110"> |<img src=https://user-images.githubusercontent.com/54695910/203604475-724be708-27d6-4e56-9c2f-ae2eca24118c.png height="100" width="110">|<img src=https://user-images.githubusercontent.com/54695910/203604459-9a2915bc-91dc-460c-bff6-a0e2584d2eff.png height="100" width="110">|<img src=https://user-images.githubusercontent.com/54695910/203604453-6ce0118e-7b93-4044-8a92-56f2ab65c26a.png height="100" width="110">|<img src=https://user-images.githubusercontent.com/54695910/203604418-7c9703b5-1805-457e-966c-5a6625f212ff.png height="100" width="110">|

- [**more releases information**](./releases)

## Contents

* <details open><summary><b>📖 Tutorials（click to fold）</b></summary><div>
  
  - Install
    
    - [How to Install FastDeploy Prebuilt Libraries](docs/en/build_and_install/download_prebuilt_libraries.md)
    - [How to Build and Install FastDeploy Library on GPU Platform](docs/en/build_and_install/gpu.md)
    - [How to Build and Install FastDeploy Library on CPU Platform](docs/en/build_and_install/cpu.md)
    - [How to Build and Install FastDeploy Library on IPU Platform](docs/en/build_and_install/ipu.md)
    - [How to Build and Install FastDeploy Library on  Nvidia Jetson Platform](docs/en/build_and_install/jetson.md)
    - [How to Build and Install FastDeploy Library on Android Platform](docs/en/build_and_install/android.md)
  
  - A Quick Start - Demos
    
    - [Python Deployment Demo](docs/en/quick_start/models/python.md)
    - [C++ Deployment Demo](docs/en/quick_start/models/cpp.md)
    - [A Quick Start on Runtime Python](docs/en/quick_start/runtime/python.md)
    - [A Quick Start on Runtime C++](docs/en/quick_start/runtime/cpp.md)
  
  - API (To be continued)
    
    - [Python API](https://baidu-paddle.github.io/fastdeploy-api/python/html/)
    - [C++ API](https://baidu-paddle.github.io/fastdeploy-api/cpp/html/)
  
  - Performance Optimization
    
    - [Quantization Acceleration](docs/en/quantize.md)
  
  - Frequent Q&As
    
    - [1. How to Change Inference Backends](docs/en/faq/how_to_change_backend.md)
    - [2. How to Use FastDeploy C++ SDK on Windows Platform](docs/en/faq/use_sdk_on_windows.md)
    - [3. How to Use FastDeploy C++ SDK on Android Platform](docs/en/faq/use_sdk_on_android.md)(To be Continued)
    - [4. Tricks of TensorRT](docs/en/faq/tensorrt_tricks.md)
    - [5. How to Develop a New Model](docs/en/faq/develop_a_new_model.md)(To be Continued)
  
  - More FastDeploy Deployment Module
    
    - [deployment AI Model as a Service](./serving)
    
    - [Benchmark Testing](./benchmark)
      
</div></details>

* **🖥️ Server-side and Cloud Deployment**
  
  * [A Quick Start for Python SDK](#fastdeploy-quick-start-python)  
  * [A Quick Start for C++ SDK](#fastdeploy-quick-start-cpp)
  * [Supported Server-side and Cloud Model List](#fastdeploy-server-models)

* **📱 Mobile and Edge Device Deployment**
  
  * [Supported Mobile and Edge Model List](#fastdeploy-edge-models)

* **🌐 Browser and Mini Program Deployment**
  
  * [Supported Web and Mini Program Model List](#fastdeploy-web-models)

* [**Community**](#fastdeploy-community)

* [**Acknowledge**](#fastdeploy-acknowledge)  

* [**License**](#fastdeploy-license)

## 🖥️ Server-side and Cloud Deployment

<div id="fastdeploy-quick-start-python"></div>

<details close>
<summary><b>A Quick Start for Python SDK（click to expand）</b></summary><div>

#### Installation

##### Prerequisites

- CUDA >= 11.2 、cuDNN >= 8.0  、 Python >= 3.6
- OS: Linux x86_64/macOS/Windows 10

##### Install FastDeploy SDK with both CPU and GPU support

```bash
pip install fastdeploy-gpu-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

##### [Conda Installation (Recommended)](docs/cn/build_and_install/download_prebuilt_libraries.md)

```bash
conda config --add channels conda-forge && conda install cudatoolkit=11.2 cudnn=8.2
```

##### Install FastDeploy SDK with only CPU support

```bash
pip install fastdeploy-python -f https://www.paddlepaddle.org.cn/whl/fastdeploy.html
```

#### Python Inference Example

* Prepare model and picture

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results

```python
# For deployment of GPU/TensorRT, please refer to examples/vision/detection/paddledetection/python
import cv2
import fastdeploy.vision as vision

im = cv2.imread("000000014439.jpg")
model = vision.detection.PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                 "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                 "ppyoloe_crn_l_300e_coco/infer_cfg.yml")

result = model.predict(im)
print(result)

vis_im = vision.vis_detection(im, result, score_threshold=0.5)
cv2.imwrite("vis_image.jpg", vis_im)
```

</div></details>

<div id="fastdeploy-quick-start-cpp"></div>

<details>
<summary><b>A Quick Start for C++ SDK（click to expand）</b></summary><div>

#### Installation

- Please refer to [C++ Prebuilt Libraries Download](docs/cn/build_and_install/download_prebuilt_libraries.md)

#### C++ Inference Example

* Prepare models and pictures

```bash
wget https://bj.bcebos.com/paddlehub/fastdeploy/ppyoloe_crn_l_300e_coco.tgz
tar xvf ppyoloe_crn_l_300e_coco.tgz
wget https://gitee.com/paddlepaddle/PaddleDetection/raw/release/2.4/demo/000000014439.jpg
```

* Test inference results

```C++
// For GPU/TensorRT deployment, please refer to examples/vision/detection/paddledetection/cpp
#include "fastdeploy/vision.h"

int main(int argc, char* argv[]) {
  namespace vision = fastdeploy::vision;
  auto im = cv::imread("000000014439.jpg");
  auto model = vision::detection::PPYOLOE("ppyoloe_crn_l_300e_coco/model.pdmodel",
                                          "ppyoloe_crn_l_300e_coco/model.pdiparams",
                                          "ppyoloe_crn_l_300e_coco/infer_cfg.yml");

  vision::DetectionResult res;
  model.Predict(&im, &res);

  auto vis_im = vision::VisDetection(im, res, 0.5);
  cv::imwrite("vis_image.jpg", vis_im);
  return 0;
 }
```

</div></details>

For more deployment models, please refer to [Vision Model Deployment Examples](examples/vision) .

<div id="fastdeploy-server-models"></div>

### Server-side and Cloud Model List🔥🔥🔥🔥🔥

Notes: ✅: already supported; ❔: to be supported in the future;  N/A: Not Available;

<details open><summary><b> Server-side and Cloud Model List（click to fold）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198620704-741523c1-dec7-44e5-9f2b-29ddd9997344.png" />
</div>

| Task                   | Model                                                                                        | Linux                                            | Linux                    | Win                      | Win                      | Mac                     | Mac                   | Linux                      | Linux                       | Linux                       | Linux   |
|:----------------------:|:--------------------------------------------------------------------------------------------:|:------------------------------------------------:|:------------------------:|:------------------------:|:------------------------:|:-----------------------:|:---------------------:|:--------------------------:|:---------------------------:|:---------------------------:|:-------:|
| ---                    | ---                                                                                          | <font size=2> X86 CPU                            | <font size=2> NVIDIA GPU | <font size=2> Intel  CPU | <font size=2> NVIDIA GPU | <font size=2> Intel CPU | <font size=2> Arm CPU | <font size=2>  AArch64 CPU | <font size=2> NVIDIA Jetson | <font size=2> Graphcore IPU | Serving |
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

## 📱 Mobile and Edge Device Deployment

<div id="fastdeploy-edge-models"></div>

### Mobile and Edge Model List 🔥🔥🔥🔥
    
<details open><summary><b> Mobile and Edge Model List（click to fold）</b></summary><div>

<div align="center">
  <img src="https://user-images.githubusercontent.com/54695910/198620704-741523c1-dec7-44e5-9f2b-29ddd9997344.png" />
</div>

| Task               | Model                                                                                     | Size (MB) | Linux   | Android | iOS     | Linux                         | Linux                                | Linux                             | Linux                             | TBD...  |
|:------------------:|:-----------------------------------------------------------------------------------------:|:---------:|:-------:|:-------:|:-------:|:-----------------------------:|:------------------------------------:|:---------------------------------:|:---------------------------------:|:-------:|
| ---                | ---                                                                                       | ---       | ARM CPU | ARM CPU | ARM CPU | Rockchip-NPU<br>RK3568/RK3588 | Rockchip-NPU<br>RV1109/RV1126/RK1808 | Amlogic-NPU <br>A311D/S905D/C308X | NXP-NPU<br>i.MX&nbsp;8M&nbsp;Plus | TBD...｜ |
| Classification     | [PaddleClas/PP-LCNet](examples/vision/classification/paddleclas)                          | 11.9      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/PP-LCNetv2](examples/vision/classification/paddleclas)                        | 26.6      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/EfficientNet](examples/vision/classification/paddleclas)                      | 31.4      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/GhostNet](examples/vision/classification/paddleclas)                          | 20.8      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV1](examples/vision/classification/paddleclas)                       | 17        | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV2](examples/vision/classification/paddleclas)                       | 14.2      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/MobileNetV3](examples/vision/classification/paddleclas)                       | 22        | ✅       | ✅       | ❔       | ❔                             | ❔                                    | ❔                                 | ❔                                 | --      |
| Classification     | [PaddleClas/ShuffleNetV2](examples/vision/classification/paddleclas)                      | 9.2       | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/SqueezeNetV1.1](examples/vision/classification/paddleclas)                    | 5         | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/Inceptionv3](examples/vision/classification/paddleclas)                       | 95.5      | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/PP-HGNet](examples/vision/classification/paddleclas)                          | 59        | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Classification     | [PaddleClas/SwinTransformer_224_win7](examples/vision/classification/paddleclas)          | 352.7     | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_coco](examples/vision/detection/paddledetection)        | 4.1       | ✅       | ✅       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-PicoDet_s_320_lcnet](examples/vision/detection/paddledetection)       | 4.9       | ✅       | ✅       | ❔       | ❔                             | ✅                                    | ✅                                 | ✅                                 | --      |
| Detection          | [PaddleDetection/CenterNet](examples/vision/detection/paddledetection)                    | 4.8       | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/YOLOv3_MobileNetV3](examples/vision/detection/paddledetection)           | 94.6      | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-YOLO_tiny_650e_coco](examples/vision/detection/paddledetection)       | 4.4       | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/SSD_MobileNetV1_300_120e_voc](examples/vision/detection/paddledetection) | 23.3      | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-YOLO_ResNet50vd](examples/vision/detection/paddledetection)           | 188.5     | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-YOLOv2_ResNet50vd](examples/vision/detection/paddledetection)         | 218.7     | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | [PaddleDetection/PP-YOLO_crn_l_300e_coco](examples/vision/detection/paddledetection)      | 209.1     | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Detection          | YOLOv5s                                                                                   | 29.3      | ❔       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Face Detection     | BlazeFace                                                                                 | 1.5       | ❔       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Face Detection     | RetinaFace                                                                                | 1.7       | ❔       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| Keypoint Detection | [PaddleDetection/PP-TinyPose](examples/vision/keypointdetection/tiny_pose)                | 5.5       | ✅       | ❔       | ❔       | ❔                             | ❔                                    | ❔                                 | ❔                                 | --      |
| Segmentation       | [PaddleSeg/PP-LiteSeg(STDC1)](examples/vision/segmentation/paddleseg)                     | 32.2      | ✅       | ❔       | ❔       | ✅                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg-Lite](examples/vision/segmentation/paddleseg)                      | 0.556     | ✅       | ❔       | ❔       | ✅                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/HRNet-w18](examples/vision/segmentation/paddleseg)                             | 38.7      | ✅       | ❔       | ❔       | ✅                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/PP-HumanSeg](examples/vision/segmentation/paddleseg)                           | 107.2     | ✅       | ❔       | ❔       | ✅                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/Unet](examples/vision/segmentation/paddleseg)                                  | 53.7      | ✅       | ❔       | ❔       | ✅                             | --                                   | --                                | --                                | --      |
| Segmentation       | [PaddleSeg/Deeplabv3](examples/vision/segmentation/paddleseg)                             | 150       | ❔       | ❔       | ❔       | ✅                             |                                      |                                   |                                   |         |
| OCR                | PaddleOCR/PP-OCRv1                                                                        | 2.3+4.4   | ❔       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| OCR                | [PaddleOCR/PP-OCRv2](examples/vision/ocr/PP-OCRv2)                                        | 2.3+4.4   | ✅       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
| OCR                | [PaddleOCR/PP-OCRv3](examples/vision/ocr/PP-OCRv3)                                        | 2.4+10.6  | ✅       | ❔       | ❔       | ❔                             | ❔                                    | ❔                                 | ❔                                 | --      |
| OCR                | PaddleOCR/PP-OCRv3-tiny                                                                   | 2.4+10.7  | ❔       | ❔       | ❔       | ❔                             | --                                   | --                                | --                                | --      |
    
</div></details>

## 🌐 Browser-based Model List

<div id="fastdeploy-web-models"></div>
    
<details open><summary><b> Browser-based Model List（click to fold）</b></summary><div>

| Task               | Model                                                                                       | [web_demo](examples/application/js/web_demo) |
|:------------------:|:-------------------------------------------------------------------------------------------:|:--------------------------------------------:|
| ---                | ---                                                                                         | [Paddle.js](examples/application/js)         |
| Detection          | [FaceDetection](examples/application/js/web_demo/src/pages/cv/detection)                    | ✅                                            |
| Detection          | [ScrewDetection](examples/application/js/web_demo/src/pages/cv/detection)                   | ✅                                            |
| Segmentation       | [PaddleSeg/HumanSeg](./examples/application/js/web_demo/src/pages/cv/segmentation/HumanSeg) | ✅                                            |
| Object Recognition | [GestureRecognition](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| Object Recognition | [ItemIdentification](examples/application/js/web_demo/src/pages/cv/recognition)             | ✅                                            |
| OCR                | [PaddleOCR/PP-OCRv3](./examples/application/js/web_demo/src/pages/cv/ocr)                   | ✅                                            |

</div></details>
    
## Community

<div id="fastdeploy-community"></div>

- If you have any question or suggestion, please give us your valuable input via GitHub Issues
- **Join Us👬：**
  - **Slack**：Join our [Slack community](https://join.slack.com/t/fastdeployworkspace/shared_invite/zt-1hhvpb279-iw2pNPwrDaMBQ5OQhO3Siw) and chat with other community members about ideas
  - **WeChat**：join our **WeChat community** and chat with other community members about ideas

<div align="center">
<img src="https://user-images.githubusercontent.com/54695910/200145290-d5565d18-6707-4a0b-a9af-85fd36d35d13.jpg"  width = "225" height = "225" />
</div>

## Acknowledge

<div id="fastdeploy-acknowledge"></div>

We sincerely appreciate the open-sourced capabilities in [EasyEdge](https://ai.baidu.com/easyedge/app/openSource) as we adopt it for the SDK generation and download in this project.

## License

<div id="fastdeploy-license"></div>

FastDeploy is provided under the [Apache-2.0](./LICENSE).
