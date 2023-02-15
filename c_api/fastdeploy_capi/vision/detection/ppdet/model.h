// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "fastdeploy_capi/fd_common.h"
#include "fastdeploy_capi/fd_type.h"
#include "fastdeploy_capi/runtime_option.h"
#include "fastdeploy_capi/vision/result.h"
#include "fastdeploy_capi/vision/detection/ppdet/base_define.h"

typedef struct FD_C_TTFNetWrapper FD_C_TTFNetWrapper;
typedef struct FD_C_RuntimeOptionWrapper FD_C_RuntimeOptionWrapper;

#ifdef __cplusplus
extern "C" {
#endif

// PPYOLOE

/** \brief Create a new FD_C_PPYOLOEWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PPYOLOEWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PPYOLOE);

/** \brief Destroy a FD_C_PPYOLOEWrapper object
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ppyoloe_wrapper pointer to FD_C_PPYOLOEWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PPYOLOE, fd_c_ppyoloe_wrapper);

// PicoDet

/** \brief Create a new FD_C_PicoDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PicoDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PicoDet);

/** \brief Destroy a FD_C_PicoDetWrapper object
 *
 * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PicoDet, fd_c_picodet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_picodet_wrapper pointer to FD_C_PicoDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PicoDet, fd_c_picodet_wrapper);


// PPYOLO

/** \brief Create a new FD_C_PPYOLOWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PPYOLOWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PPYOLO);

/** \brief Destroy a FD_C_PPYOLOWrapper object
 *
 * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ppyolo_wrapper pointer to FD_C_PPYOLOWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PPYOLO, fd_c_ppyolo_wrapper);

// YOLOv3

/** \brief Create a new FD_C_YOLOv3Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_YOLOv3Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(YOLOv3);

/** \brief Destroy a FD_C_YOLOv3Wrapper object
 *
 * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_yolov3_wrapper pointer to FD_C_YOLOv3Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(YOLOv3, fd_c_yolov3_wrapper);

// PaddleYOLOX

/** \brief Create a new FD_C_PaddleYOLOXWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOXWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOX);

/** \brief Destroy a FD_C_PaddleYOLOXWrapper object
 *
 * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolox_wrapper pointer to FD_C_PaddleYOLOXWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOX, fd_c_paddleyolox_wrapper);

// FasterRCNN

/** \brief Create a new FD_C_FasterRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_FasterRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(FasterRCNN);

/** \brief Destroy a FD_C_FasterRCNNWrapper object
 *
 * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_fasterrcnn_wrapper pointer to FD_C_FasterRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(FasterRCNN, fd_c_fasterrcnn_wrapper);

// MaskRCNN

/** \brief Create a new FD_C_MaskRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_MaskRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(MaskRCNN);

/** \brief Destroy a FD_C_MaskRCNNWrapper object
 *
 * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_maskrcnn_wrapper pointer to FD_C_MaskRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(MaskRCNN, fd_c_maskrcnn_wrapper);

// SSD

/** \brief Create a new FD_C_SSDWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_SSDWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(SSD);

/** \brief Destroy a FD_C_SSDWrapper object
 *
 * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(SSD, fd_c_ssd_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ssd_wrapper pointer to FD_C_SSDWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(SSD, fd_c_ssd_wrapper);

// PaddleYOLOv5

/** \brief Create a new FD_C_PaddleYOLOv5Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv5Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv5);

/** \brief Destroy a FD_C_PaddleYOLOv5Wrapper object
 *
 * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov5_wrapper pointer to FD_C_PaddleYOLOv5Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv5, fd_c_paddleyolov5_wrapper);

// PaddleYOLOv6

/** \brief Create a new FD_C_PaddleYOLOv6Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv6Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv6);

/** \brief Destroy a FD_C_PaddleYOLOv6Wrapper object
 *
 * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov6_wrapper pointer to FD_C_PaddleYOLOv6Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv6, fd_c_paddleyolov6_wrapper);

// PaddleYOLOv7

/** \brief Create a new FD_C_PaddleYOLOv7Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv7Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv7);

/** \brief Destroy a FD_C_PaddleYOLOv7Wrapper object
 *
 * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov7_wrapper pointer to FD_C_PaddleYOLOv7Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv7, fd_c_paddleyolov7_wrapper);

// PaddleYOLOv8

/** \brief Create a new FD_C_PaddleYOLOv8Wrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PaddleYOLOv8Wrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PaddleYOLOv8);

/** \brief Destroy a FD_C_PaddleYOLOv8Wrapper object
 *
 * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov8_wrapper pointer to FD_C_PaddleYOLOv8Wrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PaddleYOLOv8, fd_c_paddleyolov8_wrapper);

// RTMDet

/** \brief Create a new FD_C_RTMDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_RTMDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(RTMDet);

/** \brief Destroy a FD_C_RTMDetWrapper object
 *
 * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_rtmdet_wrapper pointer to FD_C_RTMDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(RTMDet, fd_c_rtmdet_wrapper);

// CascadeRCNN

/** \brief Create a new FD_C_CascadeRCNNWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_CascadeRCNNWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(CascadeRCNN);

/** \brief Destroy a FD_C_CascadeRCNNWrapper object
 *
 * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_cascadercnn_wrapper pointer to FD_C_CascadeRCNNWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(CascadeRCNN, fd_c_cascadercnn_wrapper);

// PSSDet

/** \brief Create a new FD_C_PSSDetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_PSSDetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(PSSDet);

/** \brief Destroy a FD_C_PSSDetWrapper object
 *
 * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_pssdet_wrapper pointer to FD_C_PSSDetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(PSSDet, fd_c_pssdet_wrapper);

// RetinaNet

/** \brief Create a new FD_C_RetinaNetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_RetinaNetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(RetinaNet);

/** \brief Destroy a FD_C_RetinaNetWrapper object
 *
 * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_retinanet_wrapper pointer to FD_C_RetinaNetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(RetinaNet, fd_c_retinanet_wrapper);

// TTFNetSOD

/** \brief Create a new FD_C_TTFNetSODWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_TTFNetSODWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(TTFNetSOD);

/** \brief Destroy a FD_C_TTFNetSODWrapper object
 *
 * \param[in] fd_c_paddleyolov5sod_wrapper pointer to FD_C_TTFNetSODWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(TTFNetSOD, fd_c_paddleyolov5sod_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_paddleyolov5sod_wrapper pointer to FD_C_TTFNetSODWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(TTFNetSOD, fd_c_paddleyolov5sod_wrapper);

// FCOS

/** \brief Create a new FD_C_FCOSWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_FCOSWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(FCOS);

/** \brief Destroy a FD_C_FCOSWrapper object
 *
 * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(FCOS, fd_c_fcos_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_fcos_wrapper pointer to FD_C_FCOSWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(FCOS, fd_c_fcos_wrapper);

// TTFNet

/** \brief Create a new FD_C_TTFNetWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_TTFNetWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(TTFNet);

/** \brief Destroy a FD_C_TTFNetWrapper object
 *
 * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_ttfnet_wrapper pointer to FD_C_TTFNetWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(TTFNet, fd_c_ttfnet_wrapper);

// TOOD

/** \brief Create a new FD_C_TOODWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_TOODWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(TOOD);

/** \brief Destroy a FD_C_TOODWrapper object
 *
 * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(TOOD, fd_c_tood_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_tood_wrapper pointer to FD_C_TOODWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(TOOD, fd_c_tood_wrapper);

// GFL

/** \brief Create a new FD_C_GFLWrapper object
 *
 * \param[in] model_file Path of model file, e.g resnet/model.pdmodel
 * \param[in] params_file Path of parameter file, e.g resnet/model.pdiparams, if the model format is ONNX, this parameter will be ignored
 * \param[in] config_file Path of configuration file for deployment, e.g resnet/infer_cfg.yml
 * \param[in] fd_c_runtime_option_wrapper RuntimeOption for inference, the default will use cpu, and choose the backend defined in `valid_cpu_backends`
 * \param[in] model_format Model format of the loaded model, default is Paddle format
 *
 * \return Return a pointer to FD_C_GFLWrapper object
 */

DECLARE_CREATE_WRAPPER_FUNCTION(GFL);

/** \brief Destroy a FD_C_GFLWrapper object
 *
 * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
 */

DECLARE_DESTROY_WRAPPER_FUNCTION(GFL, fd_c_gfl_wrapper);

/** \brief Predict the detection result for an input image
 *
 * \param[in] fd_c_gfl_wrapper pointer to FD_C_GFLWrapper object
 * \param[in] img pointer to cv::Mat image
 * \param[in] fd_c_detection_result_wrapper pointer to FD_C_DetectionResultWrapper object, which stores the result.
 */

DECLARE_PREDICT_FUNCTION(GFL, fd_c_gfl_wrapper);

//

#ifdef __cplusplus
}  // extern "C"
#endif
