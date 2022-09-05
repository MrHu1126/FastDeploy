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

#include "fastdeploy/backends/tensorrt/trt_backend.h"
#include <cstring>
#include "NvInferSafeRuntime.h"
#include "fastdeploy/utils/utils.h"
#ifdef ENABLE_PADDLE_FRONTEND
#include "paddle2onnx/converter.h"
#endif

namespace fastdeploy {

FDTrtLogger* FDTrtLogger::logger = nullptr;

size_t TrtDataTypeSize(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return sizeof(float);
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return sizeof(float) / 2;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return sizeof(int8_t);
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return sizeof(int32_t);
  }
  // kBOOL
  return sizeof(bool);
}

FDDataType GetFDDataType(const nvinfer1::DataType& dtype) {
  if (dtype == nvinfer1::DataType::kFLOAT) {
    return FDDataType::FP32;
  } else if (dtype == nvinfer1::DataType::kHALF) {
    return FDDataType::FP16;
  } else if (dtype == nvinfer1::DataType::kINT8) {
    return FDDataType::INT8;
  } else if (dtype == nvinfer1::DataType::kINT32) {
    return FDDataType::INT32;
  }
  // kBOOL
  return FDDataType::BOOL;
}

std::vector<int> toVec(const nvinfer1::Dims& dim) {
  std::vector<int> out(dim.d, dim.d + dim.nbDims);
  return out;
}

bool CheckDynamicShapeConfig(const paddle2onnx::OnnxReader& reader,
                             const TrtBackendOption& option) {
  // paddle2onnx::ModelTensorInfo inputs[reader.NumInputs()];
  // std::string input_shapes[reader.NumInputs()];
  std::vector<paddle2onnx::ModelTensorInfo> inputs(reader.NumInputs());
  std::vector<std::string> input_shapes(reader.NumInputs());
  for (int i = 0; i < reader.NumInputs(); ++i) {
    reader.GetInputInfo(i, &inputs[i]);

    // change 0 to -1, when input_dim is a string, onnx will make it to zero
    for (int j = 0; j < inputs[i].rank; ++j) {
      if (inputs[i].shape[j] <= 0) {
        inputs[i].shape[j] = -1;
      }
    }

    input_shapes[i] = "";
    for (int j = 0; j < inputs[i].rank; ++j) {
      if (j != inputs[i].rank - 1) {
        input_shapes[i] += (std::to_string(inputs[i].shape[j]) + ", ");
      } else {
        input_shapes[i] += std::to_string(inputs[i].shape[j]);
      }
    }
  }

  bool all_check_passed = true;
  for (int i = 0; i < reader.NumInputs(); ++i) {
    bool contain_unknown_dim = false;
    for (int j = 0; j < inputs[i].rank; ++j) {
      if (inputs[i].shape[j] < 0) {
        contain_unknown_dim = true;
      }
    }

    std::string name(inputs[i].name, strlen(inputs[i].name));
    FDINFO << "The loaded model's input tensor:" << name
           << " has shape [" + input_shapes[i] << "]." << std::endl;
    if (contain_unknown_dim) {
      auto iter1 = option.min_shape.find(name);
      auto iter2 = option.max_shape.find(name);
      auto iter3 = option.opt_shape.find(name);
      if (iter1 == option.min_shape.end() || iter2 == option.max_shape.end() ||
          iter3 == option.opt_shape.end()) {
        FDERROR << "The loaded model's input tensor:" << name
                << " has dynamic shape [" + input_shapes[i] +
                       "], but didn't configure it's shape for tensorrt with "
                       "SetTrtInputShape correctly."
                << std::endl;
        all_check_passed = false;
      }
    }
  }

  return all_check_passed;
}

bool TrtBackend::InitFromTrt(const std::string& trt_engine_file,
                             const TrtBackendOption& option) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  cudaSetDevice(option.gpu_id);

  std::ifstream fin(trt_engine_file, std::ios::binary | std::ios::in);
  if (!fin) {
    FDERROR << "Failed to open TensorRT Engine file " << trt_engine_file
            << std::endl;
    return false;
  }
  fin.seekg(0, std::ios::end);
  std::string engine_buffer;
  engine_buffer.resize(fin.tellg());
  fin.seekg(0, std::ios::beg);
  fin.read(&(engine_buffer.at(0)), engine_buffer.size());
  fin.close();
  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }
  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(engine_buffer.data(),
                                     engine_buffer.size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  FDASSERT(cudaStreamCreate(&stream_) == 0,
           "[ERROR] Error occurs while calling cudaStreamCreate().");
  GetInputOutputInfo();
  initialized_ = true;
  return true;
}

bool TrtBackend::InitFromPaddle(const std::string& model_file,
                                const std::string& params_file,
                                const TrtBackendOption& option, bool verbose) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }

#ifdef ENABLE_PADDLE_FRONTEND
  std::vector<paddle2onnx::CustomOp> custom_ops;
  for (auto& item : option.custom_op_info_) {
    paddle2onnx::CustomOp op;
    std::strcpy(op.op_name, item.first.c_str());
    std::strcpy(op.export_op_name, item.second.c_str());
    custom_ops.emplace_back(op);
  }
  char* model_content_ptr;
  int model_content_size = 0;
  if (!paddle2onnx::Export(model_file.c_str(), params_file.c_str(),
                           &model_content_ptr, &model_content_size, 11, true,
                           verbose, true, true, true, custom_ops.data(),
                           custom_ops.size())) {
    FDERROR << "Error occured while export PaddlePaddle to ONNX format."
            << std::endl;
    return false;
  }

  if (option.remove_multiclass_nms_) {
    char* new_model = nullptr;
    int new_model_size = 0;
    if (!paddle2onnx::RemoveMultiClassNMS(model_content_ptr, model_content_size,
                                          &new_model, &new_model_size)) {
      FDERROR << "Try to remove MultiClassNMS failed." << std::endl;
      return false;
    }
    delete[] model_content_ptr;
    std::string onnx_model_proto(new_model, new_model + new_model_size);
    delete[] new_model;
    return InitFromOnnx(onnx_model_proto, option, true);
  }

  std::string onnx_model_proto(model_content_ptr,
                               model_content_ptr + model_content_size);
  delete[] model_content_ptr;
  model_content_ptr = nullptr;
  return InitFromOnnx(onnx_model_proto, option, true);
#else
  FDERROR << "Didn't compile with PaddlePaddle frontend, you can try to "
             "call `InitFromOnnx` instead."
          << std::endl;
  return false;
#endif
}

bool TrtBackend::InitFromOnnx(const std::string& model_file,
                              const TrtBackendOption& option,
                              bool from_memory_buffer) {
  if (initialized_) {
    FDERROR << "TrtBackend is already initlized, cannot initialize again."
            << std::endl;
    return false;
  }
  cudaSetDevice(option.gpu_id);

  std::string onnx_content = "";
  if (!from_memory_buffer) {
    std::ifstream fin(model_file.c_str(), std::ios::binary | std::ios::in);
    if (!fin) {
      FDERROR << "[ERROR] Failed to open ONNX model file: " << model_file
              << std::endl;
      return false;
    }
    fin.seekg(0, std::ios::end);
    onnx_content.resize(fin.tellg());
    fin.seekg(0, std::ios::beg);
    fin.read(&(onnx_content.at(0)), onnx_content.size());
    fin.close();
  } else {
    onnx_content = model_file;
  }

  // This part of code will record the original outputs order
  // because the converted tensorrt network may exist wrong order of outputs
  outputs_order_.clear();
  auto onnx_reader =
      paddle2onnx::OnnxReader(onnx_content.c_str(), onnx_content.size());
  for (int i = 0; i < onnx_reader.NumOutputs(); ++i) {
    std::string name(
        onnx_reader.output_names[i],
        onnx_reader.output_names[i] + strlen(onnx_reader.output_names[i]));
    outputs_order_[name] = i;
  }
  if (!CheckDynamicShapeConfig(onnx_reader, option)) {
    FDERROR << "TrtBackend::CheckDynamicShapeConfig failed." << std::endl;
    return false;
  }

  if (option.serialize_file != "") {
    std::ifstream fin(option.serialize_file, std::ios::binary | std::ios::in);
    if (fin) {
      FDINFO << "Detect serialized TensorRT Engine file in "
             << option.serialize_file << ", will load it directly."
             << std::endl;
      fin.close();
      return InitFromTrt(option.serialize_file, option);
    }
  }

  if (!CreateTrtEngine(onnx_content, option)) {
    return false;
  }

  context_ = std::shared_ptr<nvinfer1::IExecutionContext>(
      engine_->createExecutionContext());
  FDASSERT(cudaStreamCreate(&stream_) == 0,
           "[ERROR] Error occurs while calling cudaStreamCreate().");
  GetInputOutputInfo();
  initialized_ = true;
  return true;
}

bool TrtBackend::Infer(std::vector<FDTensor>& inputs,
                       std::vector<FDTensor>* outputs) {
  AllocateBufferInDynamicShape(inputs, outputs);
  std::vector<void*> input_binds(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].dtype == FDDataType::INT64) {
      int64_t* data = static_cast<int64_t*>(inputs[i].Data());
      std::vector<int32_t> casted_data(data, data + inputs[i].Numel());
      FDASSERT(cudaMemcpyAsync(inputs_buffer_[inputs[i].name].data(),
                               static_cast<void*>(casted_data.data()),
                               inputs[i].Nbytes() / 2, cudaMemcpyHostToDevice,
                               stream_) == 0,
               "[ERROR] Error occurs while copy memory from CPU to GPU.");
    } else {
      FDASSERT(cudaMemcpyAsync(inputs_buffer_[inputs[i].name].data(),
                               inputs[i].Data(), inputs[i].Nbytes(),
                               cudaMemcpyHostToDevice, stream_) == 0,
               "[ERROR] Error occurs while copy memory from CPU to GPU.");
    }
  }
  if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
    FDERROR << "Failed to Infer with TensorRT." << std::endl;
    return false;
  }
  for (size_t i = 0; i < outputs->size(); ++i) {
    FDASSERT(cudaMemcpyAsync((*outputs)[i].Data(),
                             outputs_buffer_[(*outputs)[i].name].data(),
                             (*outputs)[i].Nbytes(), cudaMemcpyDeviceToHost,
                             stream_) == 0,
             "[ERROR] Error occurs while copy memory from GPU to CPU.");
  }
  return true;
}

void TrtBackend::GetInputOutputInfo() {
  inputs_desc_.clear();
  outputs_desc_.clear();
  auto num_binds = engine_->getNbBindings();
  for (auto i = 0; i < num_binds; ++i) {
    std::string name = std::string(engine_->getBindingName(i));
    auto shape = toVec(engine_->getBindingDimensions(i));
    auto dtype = engine_->getBindingDataType(i);
    if (engine_->bindingIsInput(i)) {
      inputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype});
      inputs_buffer_[name] = FDDeviceBuffer(dtype);
    } else {
      outputs_desc_.emplace_back(TrtValueInfo{name, shape, dtype});
      outputs_buffer_[name] = FDDeviceBuffer(dtype);
    }
  }
  bindings_.resize(num_binds);
}

void TrtBackend::AllocateBufferInDynamicShape(
    const std::vector<FDTensor>& inputs, std::vector<FDTensor>* outputs) {
  for (const auto& item : inputs) {
    auto idx = engine_->getBindingIndex(item.name.c_str());
    std::vector<int> shape(item.shape.begin(), item.shape.end());
    auto dims = ToDims(shape);
    context_->setBindingDimensions(idx, dims);
    if (item.Nbytes() > inputs_buffer_[item.name].nbBytes()) {
      inputs_buffer_[item.name].resize(dims);
      bindings_[idx] = inputs_buffer_[item.name].data();
    }
  }
  if (outputs->size() != outputs_desc_.size()) {
    outputs->resize(outputs_desc_.size());
  }
  for (size_t i = 0; i < outputs_desc_.size(); ++i) {
    auto idx = engine_->getBindingIndex(outputs_desc_[i].name.c_str());
    auto output_dims = context_->getBindingDimensions(idx);

    // find the original index of output
    auto iter = outputs_order_.find(outputs_desc_[i].name);
    FDASSERT(iter != outputs_order_.end(),
             "Cannot find output: %s of tensorrt network from the original model.", outputs_desc_[i].name.c_str());
    auto ori_idx = iter->second;
    (*outputs)[ori_idx].dtype = GetFDDataType(outputs_desc_[i].dtype);
    (*outputs)[ori_idx].shape.assign(output_dims.d,
                                     output_dims.d + output_dims.nbDims);
    (*outputs)[ori_idx].name = outputs_desc_[i].name;
    (*outputs)[ori_idx].data.resize(Volume(output_dims) *
                                    TrtDataTypeSize(outputs_desc_[i].dtype));
    if ((*outputs)[ori_idx].Nbytes() >
        outputs_buffer_[outputs_desc_[i].name].nbBytes()) {
      outputs_buffer_[outputs_desc_[i].name].resize(output_dims);
      bindings_[idx] = outputs_buffer_[outputs_desc_[i].name].data();
    }
  }
}

bool TrtBackend::CreateTrtEngine(const std::string& onnx_model,
                                 const TrtBackendOption& option) {
  const auto explicitBatch =
      1U << static_cast<uint32_t>(
          nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  builder_ = FDUniquePtr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(*FDTrtLogger::Get()));
  if (!builder_) {
    FDERROR << "Failed to call createInferBuilder()." << std::endl;
    return false;
  }
  network_ = FDUniquePtr<nvinfer1::INetworkDefinition>(
      builder_->createNetworkV2(explicitBatch));
  if (!network_) {
    FDERROR << "Failed to call createNetworkV2()." << std::endl;
    return false;
  }
  auto config = FDUniquePtr<nvinfer1::IBuilderConfig>(
      builder_->createBuilderConfig());
  if (!config) {
    FDERROR << "Failed to call createBuilderConfig()." << std::endl;
    return false;
  }

  if (option.enable_fp16) {
    if (!builder_->platformHasFastFp16()) {
      FDWARNING << "Detected FP16 is not supported in the current GPU, "
                   "will use FP32 instead."
                << std::endl;
    } else {
      config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
  }

  parser_ = FDUniquePtr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network_, *FDTrtLogger::Get()));
  if (!parser_) {
    FDERROR << "Failed to call createParser()." << std::endl;
    return false;
  }
  if (!parser_->parse(onnx_model.data(), onnx_model.size())) {
    FDERROR << "Failed to parse ONNX model by TensorRT." << std::endl;
    return false;
  }

  FDINFO << "Start to building TensorRT Engine..." << std::endl;
  bool fp16 = builder_->platformHasFastFp16();
  builder_->setMaxBatchSize(option.max_batch_size);

  config->setMaxWorkspaceSize(option.max_workspace_size);

  if (option.max_shape.size() > 0) {
    auto profile = builder_->createOptimizationProfile();
    FDASSERT(option.max_shape.size() == option.min_shape.size() &&
                 option.min_shape.size() == option.opt_shape.size(),
             "[TrtBackend] Size of max_shape/opt_shape/min_shape in "
             "TrtBackendOption should keep same.");
    for (const auto& item : option.min_shape) {
      // set min shape
      FDASSERT(profile->setDimensions(item.first.c_str(),
                                      nvinfer1::OptProfileSelector::kMIN,
                                      ToDims(item.second)),
               "[TrtBackend] Failed to set min_shape for input: %s in TrtBackend.", item.first.c_str());

      // set optimization shape
      auto iter = option.opt_shape.find(item.first);
      FDASSERT(iter != option.opt_shape.end(),
               "[TrtBackend] Cannot find input name: %s in TrtBackendOption::opt_shape.", item.first.c_str());
      FDASSERT(profile->setDimensions(item.first.c_str(),
                                      nvinfer1::OptProfileSelector::kOPT,
                                      ToDims(iter->second)),
               "[TrtBackend] Failed to set opt_shape for input: %s in TrtBackend.", item.first.c_str());
      // set max shape
      iter = option.max_shape.find(item.first);
      FDASSERT(iter != option.max_shape.end(),
               "[TrtBackend] Cannot find input name: %s in TrtBackendOption::max_shape.", item.first);
      FDASSERT(profile->setDimensions(item.first.c_str(),
                                      nvinfer1::OptProfileSelector::kMAX,
                                      ToDims(iter->second)),
               "[TrtBackend] Failed to set max_shape for input: %s in TrtBackend.", item.first);
    }
    config->addOptimizationProfile(profile);
  }

  FDUniquePtr<nvinfer1::IHostMemory> plan{
      builder_->buildSerializedNetwork(*network_, *config)};
  if (!plan) {
    FDERROR << "Failed to call buildSerializedNetwork()." << std::endl;
    return false;
  }

  FDUniquePtr<nvinfer1::IRuntime> runtime{
      nvinfer1::createInferRuntime(*FDTrtLogger::Get())};
  if (!runtime) {
    FDERROR << "Failed to call createInferRuntime()." << std::endl;
    return false;
  }

  engine_ = std::shared_ptr<nvinfer1::ICudaEngine>(
      runtime->deserializeCudaEngine(plan->data(), plan->size()),
      FDInferDeleter());
  if (!engine_) {
    FDERROR << "Failed to call deserializeCudaEngine()." << std::endl;
    return false;
  }

  FDINFO << "TensorRT Engine is built succussfully." << std::endl;
  if (option.serialize_file != "") {
    FDINFO << "Serialize TensorRTEngine to local file " << option.serialize_file
           << "." << std::endl;
    std::ofstream engine_file(option.serialize_file.c_str());
    if (!engine_file) {
      FDERROR << "Failed to open " << option.serialize_file << " to write."
              << std::endl;
      return false;
    }
    engine_file.write(static_cast<char*>(plan->data()), plan->size());
    engine_file.close();
    FDINFO << "TensorRTEngine is serialized to local file "
           << option.serialize_file
           << ", we can load this model from the seralized engine "
              "directly next time."
           << std::endl;
  }
  return true;
}

TensorInfo TrtBackend::GetInputInfo(int index) {
  FDASSERT(index < NumInputs(), "The index: %d should less than the number of inputs: %d.", index, NumInputs());
  TensorInfo info;
  info.name = inputs_desc_[index].name;
  info.shape.assign(inputs_desc_[index].shape.begin(),
                    inputs_desc_[index].shape.end());
  info.dtype = GetFDDataType(inputs_desc_[index].dtype);
  return info;
}

TensorInfo TrtBackend::GetOutputInfo(int index) {
  FDASSERT(index < NumOutputs(),
           "The index: %d should less than the number of outputs: %d.", index, NumOutputs());
  TensorInfo info;
  info.name = outputs_desc_[index].name;
  info.shape.assign(outputs_desc_[index].shape.begin(),
                    outputs_desc_[index].shape.end());
  info.dtype = GetFDDataType(outputs_desc_[index].dtype);
  return info;
}
}  // namespace fastdeploy
