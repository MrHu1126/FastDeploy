[简体中文](README_CN.md) | English

# FastDeploy Serving Deployment

## Introduction

FastDeploy builds an end-to-end serving deployment based on [Triton Inference Server](https://github.com/triton-inference-server/server). The underlying backend uses the FastDeploy high-performance Runtime module and integrates the FastDeploy pre- and post-processing modules to achieve end-to-end serving deployment. It can achieve fast deployment with easy-to-use process and excellent performance.

## Prepare the environment

### Environment requirements

- Linux
- If using a GPU image, NVIDIA Driver >= 470 is required (for older Tesla architecture GPUs, such as T4, the NVIDIA Driver can be 418.40+, 440.33+, 450.51+, 460.27+)

### Obtain Image

#### CPU Image

CPU images only support Paddle/ONNX models for serving deployment on CPUs, and supported inference backends include OpenVINO, Paddle Inference, and ONNX Runtime

```shell
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.2-cpu-only-21.10
```

#### GPU Image

GPU images support Paddle/ONNX models for serving deployment on GPU and CPU, and supported inference backends including OpenVINO, TensorRT, Paddle Inference, and ONNX Runtime

```
docker pull registry.baidubce.com/paddlepaddle/fastdeploy:1.0.2-gpu-cuda11.4-trt8.4-21.10
```

Users can also compile the image by themselves according to their own needs, referring to the following documents:

- [FastDeploy Serving Deployment Image Compilation](docs/zh_CN/compile.md)

## Other Tutorials

- [How to Prepare Serving Model Repository](docs/zh_CN/model_repository.md)
- [Serving Deployment Configuration for Runtime](docs/zh_CN/model_configuration.md)
- [Demo of Serving Deployment](docs/zh_CN/demo.md)


### Serving Deployment Demo

| Task | Model  |
|---|---|
| Classification | [PaddleClas](../examples/vision/classification/paddleclas/serving/README.md) |
| Detection | [PaddleDetection](../examples/vision/detection/paddledetection/serving/README.md) |
| Detection | [ultralytics/YOLOv5](../examples/vision/detection/yolov5/serving/README.md) |
| NLP |	[PaddleNLP/ERNIE-3.0](../examples/text/ernie-3.0/serving/README.md)|
| NLP |	[PaddleNLP/UIE](../examples/text/uie/serving/README.md)|
| Speech |	[PaddleSpeech/PP-TTS](../examples/audio/pp-tts/serving/README.md)|
| OCR |	[PaddleOCR/PP-OCRv3](../examples/vision/ocr/PP-OCRv3/serving/README.md)|
