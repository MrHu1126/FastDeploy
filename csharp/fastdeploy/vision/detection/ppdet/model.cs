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

using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Collections.Generic;
using OpenCvSharp;
using fastdeploy.types_internal_c;

namespace fastdeploy {
namespace vision {
namespace detection {

public class PPYOLOE {

  public PPYOLOE(string model_file, string params_file, string config_file,
                 RuntimeOption custom_option = null,
                 ModelFormat model_format = ModelFormat.PADDLE) {
    if (custom_option == null) {
      custom_option = new RuntimeOption();
    }
    fd_ppyoloe_wrapper =
        FD_C_CreatesPPYOLOEWrapper(model_file, params_file, config_file,
                                   custom_option.GetWrapperPtr(), model_format);
  }

  ~PPYOLOE() { FD_C_DestroyPPYOLOEWrapper(fd_ppyoloe_wrapper); }

  public DetectionResult Predict(Mat img) {
    IntPtr fd_detection_result_wrapper_ptr =
        FD_C_CreateDetectionResultWrapper();
    FD_C_PPYOLOEWrapperPredict(fd_ppyoloe_wrapper, img.CvPtr,
                               fd_detection_result_wrapper_ptr);  // predict
    IntPtr fd_detection_result_ptr = FD_C_DetectionResultWrapperGetData(
        fd_detection_result_wrapper_ptr);  // get result from wrapper
    FD_DetectionResult fd_detection_result =
        (FD_DetectionResult)Marshal.PtrToStructure(fd_detection_result_ptr,
                                                   typeof(FD_DetectionResult));
    DetectionResult detection_result =
        ConvertResult.ConvertCResultToDetectionResult(fd_detection_result);
    FD_C_DestroyDetectionResultWrapper(
        fd_detection_result_wrapper_ptr);  // free fd_detection_result_wrapper_ptr
    FD_C_DestroyDetectionResult(
        fd_detection_result_ptr);  // free fd_detection_result_ptr
    return detection_result;
  }

  // below are underlying C api
  private IntPtr fd_ppyoloe_wrapper;
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_CreatesPPYOLOEWrapper")]
  private static extern IntPtr FD_C_CreatesPPYOLOEWrapper(
      string model_file, string params_file, string config_file,
      IntPtr fd_runtime_option_wrapper, ModelFormat model_format);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyPPYOLOEWrapper")]
  private static extern void
  FD_C_DestroyPPYOLOEWrapper(IntPtr fd_ppyoloe_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_PPYOLOEWrapperPredict")]
  private static extern bool
  FD_C_PPYOLOEWrapperPredict(IntPtr fd_ppyoloe_wrapper, IntPtr img,
                             IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapper")]
  private static extern IntPtr FD_C_CreateDetectionResultWrapper();
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DestroyDetectionResultWrapper")]
  private static extern void
  FD_C_DestroyDetectionResultWrapper(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll", EntryPoint = "FD_C_DestroyDetectionResult")]
  private static extern void
  FD_C_DestroyDetectionResult(IntPtr fd_detection_result);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_DetectionResultWrapperGetData")]
  private static extern IntPtr
  FD_C_DetectionResultWrapperGetData(IntPtr fd_detection_result_wrapper);
  [DllImport("fastdeploy.dll",
             EntryPoint = "FD_C_CreateDetectionResultWrapperFromData")]
  private static extern IntPtr
  FD_C_CreateDetectionResultWrapperFromData(IntPtr fd_detection_result);
}
}
}
}