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

#include "app/base_app.h"
#include "gstreamer/utils.h"

namespace fastdeploy {
namespace streamer {

static GMutex fps_lock;
static void perf_cb(gpointer context, NvDsAppPerfStruct* str) {
  guint numf = str->num_instances;

  g_mutex_lock(&fps_lock);
  for (int i = 0; i < numf; i++) {
    std::cout << "Instance: " << i << ", FPS: " << str->fps[i]
              << ", total avg.: " << str->fps_avg[i] << std::endl;
  }
  g_mutex_unlock(&fps_lock);
}

void BaseApp::SetupPerfMeasurement() {
  if (!app_config_.enable_perf_measurement) return;

  GstElement* elem = NULL;
  auto elem_names = GetSinkElemNames(GST_BIN(pipeline_));
  for (auto& elem_name : elem_names) {
    std::cout << elem_name << std::endl;
    if (elem_name.find("nvvideoencfilesinkbin") != std::string::npos) {
      elem = gst_bin_get_by_name(GST_BIN(pipeline_), elem_name.c_str());
    }
  }
  FDASSERT(elem != NULL, "Can't find a properly sink bin in the pipeline");

  GstPad* perf_pad = gst_element_get_static_pad(elem, "sink");
  FDASSERT(perf_pad != NULL, "Unable to get sink pad");

  perf_struct_.context = nullptr;
  enable_perf_measurement(&perf_struct_, perf_pad, 1,
                          (gulong)(app_config_.perf_interval_sec), 1, perf_cb);

  gst_object_unref(perf_pad);
}
}  // namespace streamer
}  // namespace fastdeploy
