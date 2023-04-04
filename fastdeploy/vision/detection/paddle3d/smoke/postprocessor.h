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
#include "fastdeploy/vision/common/processors/transform.h"
#include "fastdeploy/vision/common/result.h"

namespace fastdeploy {
namespace vision {

namespace detection {
/*! @brief Postprocessor object for Smoke serials model.
 */
class FASTDEPLOY_DECL SmokePostprocessor {
 public:
  /** \brief Create a postprocessor instance for Smoke serials model
   */
  SmokePostprocessor();

  /** \brief Process the result of runtime and fill to Detection3DResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of detection
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<FDTensor>& tensors,
      std::vector<Detection3DResult>* results);


 protected:
  float conf_threshold_;
};

}  // namespace detection
}  // namespace vision
}  // namespace fastdeploy
