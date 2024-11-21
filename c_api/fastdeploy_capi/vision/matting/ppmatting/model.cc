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

#include "fastdeploy_capi/vision/matting/ppmatting/model.h"

#include "fastdeploy_capi/internal/types_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

FD_C_PPMattingWrapper* FD_C_CreatePPMattingWrapper(
    const char* model_file, const char* params_file, const char* config_file,
    FD_C_RuntimeOptionWrapper* fd_c_runtime_option_wrapper,
    const FD_C_ModelFormat model_format) {
  auto& runtime_option = CHECK_AND_CONVERT_FD_TYPE(RuntimeOptionWrapper,
                                                   fd_c_runtime_option_wrapper);
  FD_C_PPMattingWrapper* fd_c_ppmatting_wrapper =
      new FD_C_PPMattingWrapper();
  fd_c_ppmatting_wrapper->matting_model =
      std::unique_ptr<fastdeploy::vision::matting::PPMatting>(
          new fastdeploy::vision::matting::PPMatting(
              std::string(model_file), std::string(params_file),
              std::string(config_file), *runtime_option,
              static_cast<fastdeploy::ModelFormat>(model_format)));
  return fd_c_ppmatting_wrapper;
}

void FD_C_DestroyPPMattingWrapper(
    FD_C_PPMattingWrapper* fd_c_ppmatting_wrapper) {
  delete fd_c_ppmatting_wrapper;
}

FD_C_Bool FD_C_PPMattingWrapperPredict(
    FD_C_PPMattingWrapper* fd_c_ppmatting_wrapper, FD_C_Mat img,
    FD_C_MattingResult* fd_c_matting_result) {
  cv::Mat* im = reinterpret_cast<cv::Mat*>(img);
  auto& ppmatting = CHECK_AND_CONVERT_FD_TYPE(
      PPMattingWrapper, fd_c_ppmatting_wrapper);
  FD_C_MattingResultWrapper* fd_c_matting_result_wrapper =
      FD_C_CreateMattingResultWrapper();
  auto& matting_result = CHECK_AND_CONVERT_FD_TYPE(
      MattingResultWrapper, fd_c_matting_result_wrapper);

  bool successful = ppmatting->Predict(im, matting_result.get());
  if (successful) {
    FD_C_MattingResultWrapperToCResult(fd_c_matting_result_wrapper,
                                            fd_c_matting_result);
  }
  FD_C_DestroyMattingResultWrapper(fd_c_matting_result_wrapper);
  return successful;
}

FD_C_Bool FD_C_PPMattingWrapperInitialized(
    FD_C_PPMattingWrapper* fd_c_ppmatting_wrapper) {
  auto& ppmatting = CHECK_AND_CONVERT_FD_TYPE(
      PPMattingWrapper, fd_c_ppmatting_wrapper);
  return ppmatting->Initialized();
}




#ifdef __cplusplus
}
#endif
