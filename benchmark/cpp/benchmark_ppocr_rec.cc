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

DEFINE_string(rec_label_file, "", "Path of Recognization label file of PPOCR.");

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  // Recognition Model
  auto rec_model_file = FLAGS_model + sep + "inference.pdmodel";
  auto rec_params_file = FLAGS_model + sep + "inference.pdiparams";
  if (FLAGS_backend == "paddle_trt") {
    option.paddle_infer_option.collect_trt_shape = true;
  }
  if (FLAGS_backend == "paddle_trt" || FLAGS_backend == "trt") {
    option.trt_option.SetShape("x", {1, 3, 48, 10}, {4, 3, 48, 320},
                               {8, 3, 48, 2304});
  }
  auto model_ppocr_rec = fastdeploy::vision::ocr::Recognizer(
      rec_model_file, rec_params_file, FLAGS_rec_label_file, option);
  std::string text;
  float rec_score;
  // Run once at least
  model_ppocr_rec.Predict(im, &text, &rec_score);
  // 1. Test result diff
  std::cout << "=============== Test result diff =================\n";
  std::string text_expect = "上海斯格威铂尔大酒店";
  float res_score_expect = 0.993308;
  // Calculate diff between two results.
  auto ppocr_rec_text_diff = text.compare(text_expect);
  auto ppocr_rec_score_diff = rec_score - res_score_expect;
  std::cout << "PPOCR Rec text diff: " << ppocr_rec_text_diff << std::endl;
  std::cout << "PPOCR Rec score diff: " << abs(ppocr_rec_score_diff)
            << std::endl;
  BENCHMARK_MODEL(model_ppocr_rec,
                  model_ppocr_rec.Predict(im, &text, &rec_score));
#endif
  return 0;
}