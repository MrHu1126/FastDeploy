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

#pragma once
#include "fastdeploy/fastdeploy_model.h"
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {

namespace vision {

namespace faceid {
/*! @brief CosFace model object used when to load a CosFace model exported by IngsightFace.
 */
class FASTDEPLOY_DECL InsightFaceRecognitionModel : public FastDeployModel {
 public:
  /** \brief  Set path of model file and the configuration of runtime.
   *
   * \param[in] model_file Path of model file, e.g ./arcface.onnx
   * \param[in] params_file Path of parameter file, e.g ppyoloe/model.pdiparams, if the model format is ONNX, this parameter will be ignored
   * \param[in] custom_option RuntimeOption for inference, the default will use cpu, and choose the backend defined in "valid_cpu_backends"
   * \param[in] model_format Model format of the loaded model, default is ONNX format
   */
  InsightFaceRecognitionModel(
      const std::string& model_file, const std::string& params_file = "",
      const RuntimeOption& custom_option = RuntimeOption(),
      const ModelFormat& model_format = ModelFormat::ONNX);

  virtual std::string ModelName() const { return "deepinsight/insightface"; }

  /*! @brief
  Argument for image preprocessing step, tuple of (width, height), decide the target size after resize
  */, default (112, 112)
  std::vector<int> size;
  ///  Argument for image preprocessing step, alpha values for normalization
  std::vector<float> alpha;
  ///  Argument for image preprocessing step, beta values for normalization
  std::vector<float> beta;
  /// Argument for image preprocessing step, whether to swap the B and R channel, such as BGR->RGB, default true.
  bool swap_rb;
  /// Argument for image postprocessing step, whether to apply l2 normalize to embedding values, default false;
  bool l2_normalize;
  /** \brief Predict the face recognition result for an input image
   *
   * \param[in] im The input image data, comes from cv::imread(), is a 3-D array with layout HWC, BGR format
   * \param[in] result The output face recognition result will be writen to this structure
   * \return true if the prediction successed, otherwise false
   */
  virtual bool Predict(cv::Mat* im, FaceRecognitionResult* result);

  virtual bool Initialize();

  virtual bool Preprocess(Mat* mat, FDTensor* output);

  virtual bool Postprocess(std::vector<FDTensor>& infer_result,
                           FaceRecognitionResult* result);
};

}  // namespace faceid
}  // namespace vision
}  // namespace fastdeploy
