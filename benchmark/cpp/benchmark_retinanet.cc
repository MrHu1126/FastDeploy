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

namespace vision = fastdeploy::vision;
namespace benchmark = fastdeploy::benchmark;

DEFINE_bool(no_nms, false, "Whether the model contains nms.");

int main(int argc, char* argv[]) {
#if defined(ENABLE_BENCHMARK) && defined(ENABLE_VISION)
  // Initialization
  auto option = fastdeploy::RuntimeOption();
  if (!CreateRuntimeOption(&option, argc, argv, true)) {
    return -1;
  }
  auto im = cv::imread(FLAGS_image);
  std::unordered_map<std::string, std::string> config_info;
  benchmark::ResultManager::LoadBenchmarkConfig(FLAGS_config_path,
                                                &config_info);
  auto model_file = FLAGS_model + sep + "model.pdmodel";
  auto params_file = FLAGS_model + sep + "model.pdiparams";
  auto config_file = FLAGS_model + sep + "infer_cfg.yml";
  auto model_retinanet = vision::detection::RetinaNet(model_file, params_file,
                                                        config_file, option);
  vision::DetectionResult res;
  if (config_info["precision_compare"] == "true") {
    // Run once at least
    model_retinanet.Predict(im, &res);
    // 1. Test result diff
    std::cout << "=============== Test result diff =================\n";
    // Save result to -> disk.
    std::string det_result_path = "retinanet_result.txt";
    benchmark::ResultManager::SaveDetectionResult(res, det_result_path);
    // Load result from <- disk.
    vision::DetectionResult res_loaded;
    benchmark::ResultManager::LoadDetectionResult(&res_loaded, det_result_path);
    // Calculate diff between two results.
    auto det_diff =
        benchmark::ResultManager::CalculateDiffStatis(res, res_loaded);
    std::cout << "Boxes diff: mean=" << det_diff.boxes.mean
              << ", max=" << det_diff.boxes.max
              << ", min=" << det_diff.boxes.min << std::endl;
    std::cout << "Label_ids diff: mean=" << det_diff.labels.mean
              << ", max=" << det_diff.labels.max
              << ", min=" << det_diff.labels.min << std::endl;
    // 2. Test tensor diff
    std::cout << "=============== Test tensor diff =================\n";
    std::vector<vision::DetectionResult> batch_res;
    std::vector<fastdeploy::FDTensor> input_tensors, output_tensors;
    std::vector<cv::Mat> imgs;
    imgs.push_back(im);
    std::vector<vision::FDMat> fd_images = vision::WrapMat(imgs);

    model_retinanet.GetPreprocessor().Run(&fd_images, &input_tensors);
    input_tensors[0].name = "image";
    input_tensors[1].name = "scale_factor";
    input_tensors[2].name = "im_shape";
    input_tensors.pop_back();
    model_retinanet.Infer(input_tensors, &output_tensors);
    model_retinanet.GetPostprocessor().Run(output_tensors, &batch_res);
    // Save tensor to -> disk.
    auto& tensor_dump = output_tensors[0];
    std::string det_tensor_path = "retinanet_tensor.txt";
    benchmark::ResultManager::SaveFDTensor(tensor_dump, det_tensor_path);
    // Load tensor from <- disk.
    fastdeploy::FDTensor tensor_loaded;
    benchmark::ResultManager::LoadFDTensor(&tensor_loaded, det_tensor_path);
    // Calculate diff between two tensors.
    auto det_tensor_diff = benchmark::ResultManager::CalculateDiffStatis(
        tensor_dump, tensor_loaded);
    std::cout << "Tensor diff: mean=" << det_tensor_diff.data.mean
              << ", max=" << det_tensor_diff.data.max
              << ", min=" << det_tensor_diff.data.min << std::endl;
  }
  // Run profiling
  if (FLAGS_no_nms) {
    model_retinanet.GetPostprocessor().ApplyNMS();
  }
  BENCHMARK_MODEL(model_retinanet, model_retinanet.Predict(im, &res))
  auto vis_im = vision::VisDetection(im, res);
  cv::imwrite("vis_result.jpg", vis_im);
  std::cout << "Visualized result saved in ./vis_result.jpg" << std::endl;
#endif

  return 0;
}
