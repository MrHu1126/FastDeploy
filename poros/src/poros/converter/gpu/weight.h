/***************************************************************************
* 
* Copyright (c) 2022 Baidu, Inc.  All Rights Reserved.
* 
**************************************************************************/
/**
* @file weight.h
* @author tianjinjin@baidu.com
* @date Fri Aug 13 11:14:51 CST 2021
* @brief
**/

#pragma once

#include <string>

#include "torch/script.h"
#include "NvInfer.h"

#include "poros/engine/tensorrt_engine.h"
#include "poros/util/macros.h"

namespace baidu {
namespace mirana {
namespace poros {

struct Weights {
    nvinfer1::Weights data;
    nvinfer1::Dims kernel_shape;
    nvinfer1::Dims shape;
    int64_t inputs_num;
    int64_t outputs_num;

    Weights();
    Weights(at::Tensor tensor);
    // Weights(float val);
    // Weights(int32_t val);
    friend std::ostream& operator<<(std::ostream& os, const Weights& w);
};

inline nvinfer1::ITensor* tensor_to_const(TensorrtEngine* engine, at::Tensor t) {
    auto t_weights = Weights(t);
    auto const_layer = engine->network()->addConstant(t_weights.shape, t_weights.data);
    POROS_CHECK(const_layer, "unable to freeze tensor to constant");

    auto out = const_layer->getOutput(0);

    std::ostringstream tensor_id;
    tensor_id << reinterpret_cast<int*>(out);

    LOG(INFO) << "Freezing tensor " << tensor_id.str() << " as an IConstantLayer";
    const_layer->setName(("[Freeze Tensor " + tensor_id.str() + " ]").c_str());

    return out;
}

}  // namespace poros 
}  // namespace mirana
}  // namespace baidu
