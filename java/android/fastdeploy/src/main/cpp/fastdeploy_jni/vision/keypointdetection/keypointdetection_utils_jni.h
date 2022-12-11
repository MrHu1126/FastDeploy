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

#include <jni.h>  // NOLINT
#include "fastdeploy/vision.h"  // NOLINT
#include "fastdeploy_jni/perf_jni.h"  // NOLINT
#include "fastdeploy_jni/bitmap_jni.h"  // NOLINT

namespace fastdeploy {
namespace jni {

void RenderingKeyPointDetection(
    JNIEnv *env, const cv::Mat &c_bgr,
    const vision::KeyPointDetectionResult &c_result,
    jobject argb8888_bitmap, bool save_image,
    float conf_threshold, jstring save_path);

}  // namespace jni
}  // namespace fastdeploy
