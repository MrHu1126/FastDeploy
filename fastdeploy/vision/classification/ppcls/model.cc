// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "fastdeploy/vision/classification/ppcls/model.h"
#include "fastdeploy/vision/utils/utils.h"
#include "yaml-cpp/yaml.h"
#ifdef ENABLE_OPENCV_CUDA
#include <cuda_runtime.h>
#endif

namespace fastdeploy {
namespace vision {
namespace classification {

PaddleClasModel::PaddleClasModel(const std::string& model_file,
                                 const std::string& params_file,
                                 const std::string& config_file,
                                 const RuntimeOption& custom_option,
                                 const ModelFormat& model_format) {
  config_file_ = config_file;
  valid_cpu_backends = {Backend::ORT, Backend::OPENVINO, Backend::PDINFER,
                        Backend::LITE};
  valid_gpu_backends = {Backend::ORT, Backend::PDINFER, Backend::TRT};
  runtime_option = custom_option;
  runtime_option.model_format = model_format;
  runtime_option.model_file = model_file;
  runtime_option.params_file = params_file;
#ifdef ENABLE_OPENCV_CUDA
  cudaSetDevice(runtime_option.device_id);
  cudaStream_t stream;
  FDASSERT(cudaStreamCreate(&stream) == cudaSuccess,
           "Error occurs while creating cuda stream.");
  cuda_stream_ = reinterpret_cast<void*>(stream);
  runtime_option.SetExternalStream(cuda_stream_);
#endif
  initialized = Initialize();
}

bool PaddleClasModel::Initialize() {
  reused_input_tensors.resize(1);
  reused_output_tensors.resize(1);

  if (!BuildPreprocessPipelineFromConfig()) {
    FDERROR << "Failed to build preprocess pipeline from configuration file."
            << std::endl;
    return false;
  }
  if (!InitRuntime()) {
    FDERROR << "Failed to initialize fastdeploy backend." << std::endl;
    return false;
  }
  return true;
}

PaddleClasModel::~PaddleClasModel() {
#ifdef ENABLE_OPENCV_CUDA
  if (use_cuda_preprocessing_) {
    FDASSERT(cudaStreamDestroy(reinterpret_cast<cudaStream_t>(cuda_stream_)) == cudaSuccess,
             "Error occurs while destroying cuda stream.");
  }
#endif
}

bool PaddleClasModel::BuildPreprocessPipelineFromConfig() {
  processors_.clear();
  YAML::Node cfg;
  try {
    cfg = YAML::LoadFile(config_file_);
  } catch (YAML::BadFile& e) {
    FDERROR << "Failed to load yaml file " << config_file_
            << ", maybe you should check this file." << std::endl;
    return false;
  }
  auto preprocess_cfg = cfg["PreProcess"]["transform_ops"];
  processors_.push_back(std::make_shared<BGR2RGB>());
  for (const auto& op : preprocess_cfg) {
    FDASSERT(op.IsMap(),
             "Require the transform information in yaml be Map type.");
    auto op_name = op.begin()->first.as<std::string>();
    if (op_name == "ResizeImage") {
      int target_size = op.begin()->second["resize_short"].as<int>();
      bool use_scale = false;
      int interp = cv::INTER_LINEAR;
      processors_.push_back(
          std::make_shared<ResizeByShort>(target_size, interp, use_scale));
    } else if (op_name == "CropImage") {
      int width = op.begin()->second["size"].as<int>();
      int height = op.begin()->second["size"].as<int>();
      processors_.push_back(std::make_shared<CenterCrop>(width, height));
    } else if (op_name == "NormalizeImage") {
      auto mean = op.begin()->second["mean"].as<std::vector<float>>();
      auto std = op.begin()->second["std"].as<std::vector<float>>();
      auto scale = op.begin()->second["scale"].as<float>();
      FDASSERT((scale - 0.00392157) < 1e-06 && (scale - 0.00392157) > -1e-06,
               "Only support scale in Normalize be 0.00392157, means the pixel "
               "is in range of [0, 255].");
      processors_.push_back(std::make_shared<Normalize>(mean, std));
    } else if (op_name == "ToCHWImage") {
      processors_.push_back(std::make_shared<HWC2CHW>());
    } else {
      FDERROR << "Unexcepted preprocess operator: " << op_name << "."
              << std::endl;
      return false;
    }
  }
  FuseTransforms(&processors_);
  return true;
}

void PaddleClasModel::UseCudaPreprocessing() {
#ifdef ENABLE_OPENCV_CUDA
  use_cuda_preprocessing_ = true;
  for (size_t i = 0; i < processors_.size(); ++i) {
    processors_[i]->SetCudaStream(cuda_stream_);
  }
#else
  FDWARNING << "The FastDeploy didn't compile with ENABLE_OPENCV_CUDA=ON."
            << std::endl;
  use_cuda_preprocessing_ = false;
#endif
}

bool PaddleClasModel::Preprocess(Mat* mat, FDTensor* output) {
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  int channel = mat->Channels();
  int width = mat->Width();
  int height = mat->Height();
  output->name = InputInfoOfRuntime(0).name;
  output->SetExternalData({1, channel, height, width}, FDDataType::FP32,
                          mat->GetOpenCVMat()->ptr());
  return true;
}

bool PaddleClasModel::CudaPreprocess(Mat* mat, FDTensor* output) {
#ifdef ENABLE_OPENCV_CUDA
  for (size_t i = 0; i < processors_.size(); ++i) {
    if (!(*(processors_[i].get()))(mat, ProcLib::OPENCVCUDA)) {
      FDERROR << "Failed to process image data in " << processors_[i]->Name()
              << "." << std::endl;
      return false;
    }
  }

  int channel = mat->Channels();
  int width = mat->Width();
  int height = mat->Height();
  output->name = InputInfoOfRuntime(0).name;
  output->SetExternalData({1, channel, height, width}, FDDataType::FP32,
                          mat->GetOpenCVCudaMat()->ptr(), Device::GPU);
  return true;
#else
  FDERROR << "The FastDeploy didn't compile with ENABLE_OPENCV_CUDA=ON."
          << std::endl;
  return false;
#endif
}

bool PaddleClasModel::Postprocess(const FDTensor& infer_result,
                                  ClassifyResult* result, int topk) {
  int num_classes = infer_result.shape[1];
  const float* infer_result_buffer =
      reinterpret_cast<const float*>(infer_result.Data());
  topk = std::min(num_classes, topk);
  result->label_ids =
      utils::TopKIndices(infer_result_buffer, num_classes, topk);
  result->scores.resize(topk);
  for (int i = 0; i < topk; ++i) {
    result->scores[i] = *(infer_result_buffer + result->label_ids[i]);
  }
  return true;
}

bool PaddleClasModel::Predict(cv::Mat* im, ClassifyResult* result, int topk) {
  Mat mat(*im);
  bool ret;
  if (use_cuda_preprocessing_) {
    ret = CudaPreprocess(&mat, &(reused_input_tensors[0]));
  } else {
    ret = Preprocess(&mat, &(reused_input_tensors[0]));
  }
  if (!ret) {
    FDERROR << "Failed to preprocess input data while using model:"
            << ModelName() << "." << std::endl;
    return false;
  }

  if (!Infer()) {
    FDERROR << "Failed to inference while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }

  if (!Postprocess(reused_output_tensors[0], result, topk)) {
    FDERROR << "Failed to postprocess while using model:" << ModelName() << "."
            << std::endl;
    return false;
  }
  return true;
}

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
