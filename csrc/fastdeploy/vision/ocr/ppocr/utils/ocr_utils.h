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

#include <set>
#include <vector>
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/utils/utils.h"
#include "fastdeploy/vision/common/result.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"

namespace fastdeploy {
namespace vision {
namespace ocr {

cv::Mat GetRotateCropImage(const cv::Mat &srcimage,
                           const std::vector<std::vector<int>> &box);

void cls_resize_img(const cv::Mat &img, cv::Mat &resize_img,
                    const std::vector<int> &rec_image_shape);

void crnn_resize_img(const cv::Mat &img, cv::Mat &resize_img, float wh_ratio,
                     const std::vector<int> &rec_image_shape);

}  // namespace ocr
}  // namespace vision
}  // namespace fastdeploy
