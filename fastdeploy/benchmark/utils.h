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

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace benchmark {

class FASTDEPLOY_DECL ResourceUsageMonitor {
 public:
  explicit ResourceUsageMonitor(int sampling_interval_ms, int gpu_id = 0);
  ~ResourceUsageMonitor() { StopInternal(); }

  void Start();
  void Stop();
  ResourceUsageMonitor(ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&) = delete;
  ResourceUsageMonitor(ResourceUsageMonitor&&) = delete;
  ResourceUsageMonitor& operator=(const ResourceUsageMonitor&&) = delete;

 private:
  void StopInternal();

  bool is_supported_ = false;
  bool stop_signal_ = false;
  const int sampling_interval_;
  const std::string cpu_mem_file_name_ = "result_cpu.txt";
  const std::string gpu_mem_file_name_ = "result_gpu.txt";
  const int gpu_id_ = 0;
  std::unique_ptr<std::thread> check_memory_thd_ = nullptr;
};

// Record current cpu memory usage into file
FASTDEPLOY_DECL void DumpCurrentCpuMemoryUsage(const std::string& name);

// Record current gpu memory usage into file
FASTDEPLOY_DECL void DumpCurrentGpuMemoryUsage(const std::string& name,
                                               int device_id);

// Get Max cpu memory usage
FASTDEPLOY_DECL float GetCpuMemoryUsage(const std::string& name);

// Get Max gpu memory usage
FASTDEPLOY_DECL float GetGpuMemoryUsage(const std::string& name);

}  // namespace benchmark
}  // namespace fastdeploy
