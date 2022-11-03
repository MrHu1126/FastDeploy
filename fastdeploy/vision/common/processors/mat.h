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
#include "fastdeploy/core/fd_tensor.h"
#include "fastdeploy/vision/common/processors/utils.h"
#include "opencv2/core/core.hpp"

namespace fastdeploy {
namespace vision {

enum class FASTDEPLOY_DECL ProcLib { DEFAULT, OPENCV, FLYCV };
enum Layout { HWC, CHW };

FASTDEPLOY_DECL std::ostream& operator<<(std::ostream& out, const ProcLib& p);

struct FASTDEPLOY_DECL Mat {
  explicit Mat(cv::Mat& mat) {
    cpu_mat = mat;
    layout = Layout::HWC;
    height = cpu_mat.rows;
    width = cpu_mat.cols;
    channels = cpu_mat.channels();
    mat_type = ProcLib::OPENCV;
  }

#ifdef ENABLE_FLYCV
  explicit Mat(fcv::Mat& mat) {
    fcv_mat = mat;
    layout = Layout::HWC;
    height = fcv_mat.height();
    width = fcv_mat.width();
    channels = fcv_mat.channels();
    mat_type = ProcLib::FLYCV;
  }
#endif

  // Careful if you use this interface
  // this only used if you don't want to write
  // the original data, and write to a new cv::Mat
  // then replace the old cv::Mat of this structure
  void SetMat(const cv::Mat& mat) {
    cpu_mat = mat;
    mat_type = ProcLib::OPENCV;
  }

  cv::Mat* GetOpenCVMat() {
    if (mat_type == ProcLib::OPENCV) {
      return &cpu_mat;
    } else if (mat_type == ProcLib::FLYCV) {
#ifdef ENABLE_FLYCV
      // Just a reference to fcv_mat, zero copy. After you
      // call this method, cpu_mat and fcv_mat will point
      // to the same memory buffer.
      cpu_mat = ConvertFlyCVMatToOpenCV(fcv_mat);
      mat_type = ProcLib::OPENCV;
      return &cpu_mat;
#else
      FDASSERT(false, "FastDeploy didn't compiled with FlyCV!");
#endif
    } else {
      FDASSERT(false, "The mat_type of custom Mat can not be ProcLib::DEFAULT");
    }
  }

  const cv::Mat* GetOpenCVMat() const {
    if (mat_type == ProcLib::OPENCV) {
      return &cpu_mat;
    } else if (mat_type == ProcLib::FLYCV) {
      FDASSERT(
          false,
          "Can not get OpenCV mat from const Mat if the ProcLib is FLYCV!");
    } else {
      FDASSERT(false, "The mat_type of custom Mat can not be ProcLib::DEFAULT");
    }
  }

#ifdef ENABLE_FLYCV
  void SetMat(const fcv::Mat& mat) {
    fcv_mat = mat;
    mat_type = ProcLib::FLYCV;
  }

  fcv::Mat* GetFlyCVMat() {
    if (mat_type == ProcLib::FLYCV) {
      return &fcv_mat;
    } else if (mat_type == ProcLib::OPENCV) {
      // Just a reference to cpu_mat, zero copy. After you
      // call this method, fcv_mat and cpu_mat will point
      // to the same memory buffer.
      fcv_mat = ConvertOpenCVMatToFlyCV(cpu_mat);
      mat_type = ProcLib::FLYCV;
      return &fcv_mat;
    } else {
      FDASSERT(false, "The mat_type of custom Mat can not be ProcLib::DEFAULT");
    }
  }

  const fcv::Mat* GetFlyCVMat() const {
    if (mat_type == ProcLib::FLYCV) {
      return &fcv_mat;
    } else if (mat_type == ProcLib::OPENCV) {
      FDASSERT(
          false,
          "Can not get FlyCV mat from const Mat if the ProcLib is OPENCV!");
    } else {
      FDASSERT(false,
               "The mat_type of custom Mat can not be ProcLib::DEFAULT ");
    }
  }
#endif

  void* Data();

 private:
  int channels;
  int height;
  int width;
  cv::Mat cpu_mat;
#ifdef ENABLE_FLYCV
  fcv::Mat fcv_mat;
#endif

 public:
  FDDataType Type();
  int Channels() const { return channels; }
  int Width() const { return width; }
  int Height() const { return height; }
  void SetChannels(int s) { channels = s; }
  void SetWidth(int w) { width = w; }
  void SetHeight(int h) { height = h; }

  // Transfer the vision::Mat to FDTensor
  void ShareWithTensor(FDTensor* tensor);
  // Only support copy to cpu tensor now
  bool CopyToTensor(FDTensor* tensor);

  // Debug functions
  // TODO(jiangjiajun) Develop a right process pipeline with c++
  // is not a easy things, Will add more debug function here to
  // help debug processed image. This function will print shape
  // and mean of each channels of the Mat
  void PrintInfo(const std::string& flag);

  ProcLib mat_type = ProcLib::OPENCV;
  Layout layout = Layout::HWC;
};

}  // namespace vision
}  // namespace fastdeploy
