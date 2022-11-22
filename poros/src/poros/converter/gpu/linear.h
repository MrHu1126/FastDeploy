/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file linear.h
* @author tianjinjin@baidu.com
* @date Fri Aug 20 17:21:44 CST 2021
* @brief 
**/

#pragma once

#include <string>

//from pytorch
#include "torch/script.h"

#include "poros/converter/gpu/gpu_converter.h"
#include "poros/engine/tensorrt_engine.h"

namespace baidu {
namespace mirana {
namespace poros {

class LinearConverter : public GpuConverter {
public:
    LinearConverter() {}
    virtual ~LinearConverter() {}

    //当前使用的版本
    //参考了pytorch的实现，根据情况将其转换成addmm或者matmul + add。
    bool converter(TensorrtEngine* engine, const torch::jit::Node *node);

    //DEPRECATED: 调用addFullyConnected进行组网的版本
    //在transform的模型中遇到了dimention不一致的问题,先搁置。
    bool converter_fully_connect_version(TensorrtEngine* engine, const torch::jit::Node *node);

    virtual const std::vector<std::string> schema_string() {
        return {"aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor"};
    }

    virtual const std::vector<torch::jit::NodeKind> node_kind() {
        return {torch::jit::aten::linear};
    }

};

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu