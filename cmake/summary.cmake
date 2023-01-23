# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

function(fastdeploy_summary)
  message(STATUS "")
  message(STATUS "*************FastDeploy Building Summary**********")
  message(STATUS "  CMake version             : ${CMAKE_VERSION}")
  message(STATUS "  CMake command             : ${CMAKE_COMMAND}")
  message(STATUS "  System                    : ${CMAKE_SYSTEM_NAME}")
  message(STATUS "  C++ compiler              : ${CMAKE_CXX_COMPILER}")
  message(STATUS "  C++ compiler version      : ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "  CXX flags                 : ${CMAKE_CXX_FLAGS}")
  message(STATUS "  Build type                : ${CMAKE_BUILD_TYPE}")
  get_directory_property(tmp DIRECTORY ${PROJECT_SOURCE_DIR} COMPILE_DEFINITIONS)
  message(STATUS "  Compile definitions       : ${tmp}")
  message(STATUS "  CMAKE_PREFIX_PATH         : ${CMAKE_PREFIX_PATH}")
  message(STATUS "  CMAKE_INSTALL_PREFIX      : ${CMAKE_INSTALL_PREFIX}")
  message(STATUS "  CMAKE_MODULE_PATH         : ${CMAKE_MODULE_PATH}")
  message(STATUS "")
  message(STATUS "  FastDeploy version        : ${FASTDEPLOY_VERSION}")
  message(STATUS "  ENABLE_ORT_BACKEND        : ${ENABLE_ORT_BACKEND}")
  message(STATUS "  ENABLE_RKNPU2_BACKEND     : ${ENABLE_RKNPU2_BACKEND}")
  message(STATUS "  ENABLE_SOPHGO_BACKEND     : ${ENABLE_SOPHGO_BACKEND}")
  message(STATUS "  ENABLE_PADDLE_BACKEND     : ${ENABLE_PADDLE_BACKEND}")
  message(STATUS "  ENABLE_LITE_BACKEND       : ${ENABLE_LITE_BACKEND}")
  message(STATUS "  ENABLE_POROS_BACKEND      : ${ENABLE_POROS_BACKEND}")
  message(STATUS "  ENABLE_TRT_BACKEND        : ${ENABLE_TRT_BACKEND}")
  message(STATUS "  ENABLE_OPENVINO_BACKEND   : ${ENABLE_OPENVINO_BACKEND}")
  message(STATUS "  WITH_GPU                  : ${WITH_GPU}")
  message(STATUS "  WITH_ASCEND               : ${WITH_ASCEND}")
  message(STATUS "  WITH_TIMVX                : ${WITH_TIMVX}")
  message(STATUS "  WITH_KUNLUNXIN            : ${WITH_KUNLUNXIN}")
  if(ENABLE_ORT_BACKEND)
    message(STATUS "  ONNXRuntime version       : ${ONNXRUNTIME_VERSION}")
  endif()
  if(ENABLE_PADDLE_BACKEND)
    message(STATUS "  Paddle Inference version  : ${PADDLEINFERENCE_VERSION}")
  endif()
  if(ENABLE_POROS_BACKEND)
    message(STATUS "  Poros version  : ${POROS_VERSION}")
  endif()
  if(ENABLE_OPENVINO_BACKEND)
    message(STATUS "  OpenVINO version          : ${OPENVINO_VERSION}")
  endif()
  if(WITH_GPU)
    message(STATUS "  CUDA_DIRECTORY            : ${CUDA_DIRECTORY}")
    message(STATUS "  TRT_DRECTORY              : ${TRT_DIRECTORY}")
  endif()
  message(STATUS "  ENABLE_VISION             : ${ENABLE_VISION}")
  message(STATUS "  ENABLE_TEXT               : ${ENABLE_TEXT}")
  message(STATUS "  ENABLE_ENCRYPTION         : ${ENABLE_ENCRYPTION}")
  if(ANDROID)
    message(STATUS "  ANDROID_ABI               : ${ANDROID_ABI}")
    message(STATUS "  ANDROID_PLATFORM          : ${ANDROID_PLATFORM}")
    message(STATUS "  ANDROID_NDK               : ${ANDROID_NDK}")
    message(STATUS "  ANDROID_NDK_VERSION       : ${CMAKE_ANDROID_NDK_VERSION}")
    message(STATUS "  WITH_OPENCV_STATIC:       : ${WITH_OPENCV_STATIC}")
    message(STATUS "  WITH_FLYCV_STATIC:        : ${WITH_FLYCV_STATIC}")
    if(ENABLE_LITE_BACKEND)
      message(STATUS "  WITH_LITE_STATIC          : ${WITH_LITE_STATIC}")
    endif()
    message(STATUS "  WITH_OPENMP:              : ${WITH_OPENMP}")
    message(STATUS "  WITH_JAVA:                : ${WITH_JAVA}")
  endif() 
  if (${BUILD_FASTDEPLOY_PYTHON})
    message(STATUS "  Python executable         : ${PYTHON_EXECUTABLE}")
    message(STATUS "  Python includes           : ${PYTHON_INCLUDE_DIR}")
  endif()
endfunction()
