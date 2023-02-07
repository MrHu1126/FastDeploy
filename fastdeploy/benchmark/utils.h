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

#include <sys/types.h>
#include <unistd.h>
#include <cmath>

#include "fastdeploy/utils/utils.h"

namespace fastdeploy {
namespace benchmark {

// Remove the ch characters at both ends of str
std::string strip(const std::string& str, char ch = ' ') {
  int i = 0;
  while (str[i] == ch) {
    i++;
  }
  int j = str.size() - 1;
  while (str[j] == ch) {
    j--;
  }
  return str.substr(i, j + 1 - i);
}

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
