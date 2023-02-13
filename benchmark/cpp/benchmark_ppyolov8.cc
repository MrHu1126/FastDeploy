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

#include "flags.h"
#include "macros.h"
#include "option.h"

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

int main(int argc, char* argv[]) {
  google::ParseCommandLineFlags(&argc, &argv, true);
  auto im = cv::imread(FLAGS_image);
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option)) {
    PrintUsage();
    return false;
  }
  auto model_file = FLAGS_model + sep + "model.pdmodel";
  auto params_file = FLAGS_model + sep + "model.pdiparams";
  auto config_file = FLAGS_model + sep + "infer_cfg.yml";
  auto model_ppyolov8 = fastdeploy::vision::detection::PaddleYOLOv8(
      model_file, params_file, config_file, option);
  fastdeploy::vision::DetectionResult res;
  BENCHMARK_MODEL(model_ppyolov8, model_ppyolov8.Predict(im, &res))
  auto vis_im = fastdeploy::vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
  return 0;
}