# PaddleOCR 模型部署

## PaddleOCR为多个模型组合串联任务，包含如下几个模型构成

* 文本检测 `DBDetector`
* [可选]方向分类 `Classifer` 用于调整进入文字识别前的图像方向
* 文字识别 `Recognizer` 用于从图像中识别出文字

根据不同场景, FastDeploy汇总提供如下OCR任务部署, 用户需同时下载3个模型与字典文件（或2个，分类器可选), 完成OCR整个预测流程

## PP-OCR 中英文系列模型

下表中的模型下载链接由PaddleOCR模型库提供, 详见[PP-OCR系列模型列表](https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md)

| OCR版本              | 文本框检测                                                                                                          | 方向分类模型                                                                                                         | 文字识别                                                                                                           | 字典文件                                                                              | 说明                                                      |
|:-------------------|:---------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------|:----------------------------------------------------------------------------------|:--------------------------------------------------------|
| ch_PP-OCRv3[推荐]    | [ch_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar)                  | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar)                  | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv3系列原始超轻量模型，支持中英文、多语种文本检测                            |
| en_PP-OCRv3[推荐]    | [en_PP-OCRv3_det](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar)                  | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [en_PP-OCRv3_rec](https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar)                  | [en_dict.txt](https://bj.bcebos.com/paddlehub/fastdeploy/en_dict.txt)             | OCRv3系列原始超轻量模型，支持英文与数字识别，除检测模型和识别模型的训练数据与中文模型不同以外，无其他区别 |
| ch_PP-OCRv2        | [ch_PP-OCRv2_det](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_det_infer.tar)                  | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_PP-OCRv2_rec](https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar)                  | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测                            |
| ch_PP-OCRv2_mobile | [ch_ppocr_mobile_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_mobile_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2系列原始超轻量模型，支持中英文、多语种文本检测,比PPOCRv2更加轻量               |
| ch_PP-OCRv2_server | [ch_ppocr_server_v2.0_det](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_det_infer.tar) | [ch_ppocr_mobile_v2.0_cls](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar) | [ch_ppocr_server_v2.0_rec](https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_server_v2.0_rec_infer.tar) | [ppocr_keys_v1.txt](https://bj.bcebos.com/paddlehub/fastdeploy/ppocr_keys_v1.txt) | OCRv2服务器系列模型, 支持中英文、多语种文本检测，比超轻量模型更大，但效果更好              |

## 模型转换

在RKNPU2上使用PPOCR时，我们需要把Paddle静态图模型转为RKNN模型。

### 静态图模型转RKNN格式模型

rknn_toolkit2工具暂不支持直接从Paddle静态图模型直接转换为RKNN模型，因此我们需要先将Paddle静态图模型转为RKNN模型。

```bash
# 下载模型和字典文件
wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
tar -xvf ch_PP-OCRv3_det_infer.tar

wget https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar
tar -xvf ch_ppocr_mobile_v2.0_cls_infer.tar

wget https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
tar -xvf ch_PP-OCRv3_rec_infer.tar

# 转换模型到ONNX格式的模型
paddle2onnx --model_dir ch_PP-OCRv3_det_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
            --enable_dev_version True
paddle2onnx --model_dir ch_ppocr_mobile_v2.0_cls_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
            --enable_dev_version True
paddle2onnx --model_dir ch_PP-OCRv3_rec_infer \
            --model_filename inference.pdmodel \
            --params_filename inference.pdiparams \
            --save_file ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
            --enable_dev_version True

# 固定模型的输入shape
python -m paddle2onnx.optimize --input_model ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                               --output_model ch_PP-OCRv3_det_infer/ch_PP-OCRv3_det_infer.onnx \
                               --input_shape_dict "{'x':[1,3,960,960]}"
python -m paddle2onnx.optimize --input_model ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                               --output_model ch_ppocr_mobile_v2.0_cls_infer/ch_ppocr_mobile_v2.0_cls_infer.onnx \
                               --input_shape_dict "{'x':[1,3,48,192]}"
python -m paddle2onnx.optimize --input_model ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                               --output_model ch_PP-OCRv3_rec_infer/ch_PP-OCRv3_rec_infer.onnx \
                               --input_shape_dict "{'x':[1,3,48,320]}"

# 转换ONNX模型到RKNN模型
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_det.yaml \
                              --target_platform rk3588
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_rec.yaml \
                              --target_platform rk3588
python tools/rknpu2/export.py --config_path tools/rknpu2/config/ppocrv3_cls.yaml \
                              --target_platform rk3588
```
