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

set(COMPRESSED_SUFFIX ".tgz")

if(NOT OPENCV_FILENAME)
  if(WIN32)
    if(NOT CMAKE_CL_64)
      set(OPENCV_FILENAME "opencv-win-x86-3.4.16")
    else()
      set(OPENCV_FILENAME "opencv-win-x64-3.4.16")
    endif()
    set(COMPRESSED_SUFFIX ".zip")
  elseif(APPLE)
    if(CURRENT_OSX_ARCH MATCHES "arm64")
      set(OPENCV_FILENAME "opencv-osx-arm64-3.4.16")
    else()
      set(OPENCV_FILENAME "opencv-osx-x86_64-3.4.16")
    endif()
  elseif(ANDROID)
    # Use different OpenCV libs according to toolchain
    # gcc: OpenCV 3.x, clang: OpenCV 4.x
    if(ANDROID_TOOLCHAIN MATCHES "clang")
      set(OPENCV_FILENAME "opencv-android-4.6.0")
      set(OPENCV_ANDROID_SHARED_LIB_NAME "libopencv_java4.so")
    elseif(ANDROID_TOOLCHAIN MATCHES "gcc")
      set(OPENCV_FILENAME "opencv-android-3.4.16")
      set(OPENCV_ANDROID_SHARED_LIB_NAME "libopencv_java3.so")
    else()
      message(FATAL_ERROR "Only support clang/gcc toolchain, but found ${ANDROID_TOOLCHAIN}.")
    endif()  
  elseif(IOS)
    message(FATAL_ERROR "Not support cross compiling for IOS now!")
  # Linux
  else()
    if(CMAKE_HOST_SYSTEM_PROCESSOR MATCHES "aarch64")
      set(OPENCV_FILENAME "opencv-linux-aarch64-3.4.14")
    endif()
  endif()
endif()
if(NOT OPENCV_FILENAME)
  set(OPENCV_FILENAME "opencv-linux-x64-3.4.16")
endif()

set(OPENCV_INSTALL_DIR ${THIRD_PARTY_PATH}/install/)
if(ANDROID)
  set(OPENCV_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
elseif(WIN32)
  if(NOT CMAKE_CL_64)
    set(OPENCV_URL_PREFIX "https://bj.bcebos.com/fastdeploy/third_libs")
  else()
    set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")
  endif()
else() # TODO: use fastdeploy/third_libs instead.
  set(OPENCV_URL_PREFIX "https://bj.bcebos.com/paddle2onnx/libs")
endif()
if(NOT OPENCV_URL)
  set(OPENCV_URL ${OPENCV_URL_PREFIX}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX})
endif()


if(BUILD_ON_JETSON)
  if(EXISTS /usr/lib/aarch64-linux-gnu/cmake/opencv4/)
    set(OPENCV_DIRECTORY /usr/lib/aarch64-linux-gnu/cmake/opencv4/)
  endif()
endif()

function(remove_redundant_opencv_android_abi_libraries OPENCV_JNI_DIR)
  set(ALL_OPENCV_ANDROID_ABI "arm64-v8a" "armeabi-v7a" "x86" "x86_64" "mips" "mips64" "armeabi")
  get_filename_component(OPENCV_NATIVE_DIR ${OPENCV_JNI_DIR} DIRECTORY)
  get_filename_component(OPENCV_SDK_DIR ${OPENCV_NATIVE_DIR} DIRECTORY)
  set(OPENCV_ANDROID_ETC_DIR ${OPENCV_SDK_DIR}/etc)
  set(OPENCV_ANDROID_JAVA_DIR ${OPENCV_SDK_DIR}/java)
  set(OPENCV_ANDROID_LIBS_DIR ${OPENCV_NATIVE_DIR}/libs)
  set(OPENCV_ANDROID_STATICLIBS_DIR ${OPENCV_NATIVE_DIR}/staticlibs)
  set(OPENCV_ANDROID_3RDPARTY_DIR ${OPENCV_NATIVE_DIR}/3rdparty)
  # remove redundant content
  file(REMOVE_RECURSE ${OPENCV_ANDROID_ETC_DIR})
  file(REMOVE_RECURSE ${OPENCV_ANDROID_JAVA_DIR})
  if(WITH_OPENCV_STATIC)
    foreach(_ocv_android_abi ${ALL_OPENCV_ANDROID_ABI})
      if(NOT ${_ocv_android_abi} MATCHES ${ANDROID_ABI})
        set(_curr_abi_libs_dir ${OPENCV_ANDROID_LIBS_DIR}/${_ocv_android_abi})
        set(_curr_abi_staticlibs_dir ${OPENCV_ANDROID_STATICLIBS_DIR}/${_ocv_android_abi})
        set(_curr_abi_3rdparty_dir ${OPENCV_ANDROID_3RDPARTY_DIR}/libs/${_ocv_android_abi})
        if(EXISTS ${_curr_abi_libs_dir})
          file(REMOVE_RECURSE ${_curr_abi_libs_dir})
        endif()
        if(EXISTS ${_curr_abi_staticlibs_dir})
          file(REMOVE_RECURSE ${_curr_abi_staticlibs_dir})
        endif()
        if(EXISTS ${_curr_abi_3rdparty_dir})
          file(REMOVE_RECURSE ${_curr_abi_3rdparty_dir})
        endif()  
      endif()
    endforeach()
  else()
    file(REMOVE_RECURSE ${OPENCV_ANDROID_STATICLIBS_DIR})
    file(REMOVE_RECURSE ${OPENCV_ANDROID_3RDPARTY_DIR})
    foreach(_ocv_android_abi ${ALL_OPENCV_ANDROID_ABI})
      if(NOT ${_ocv_android_abi} MATCHES ${ANDROID_ABI})
        set(_curr_abi_libs_dir ${OPENCV_ANDROID_LIBS_DIR}/${_ocv_android_abi})
        if(EXISTS ${_curr_abi_libs_dir})
          file(REMOVE_RECURSE ${_curr_abi_libs_dir})
        endif()
      endif()
    endforeach()
  endif()
endfunction()

if(OPENCV_DIRECTORY)
  message(STATUS "Use the opencv lib specified by user. The OpenCV path: ${OPENCV_DIRECTORY}")
  STRING(REGEX REPLACE "\\\\" "/" OPENCV_DIRECTORY ${OPENCV_DIRECTORY})
  # For Android, the custom path to OpenCV with JNI should look like: 
  # -DOPENCV_DIRECTORY=your-path-to/OpenCV-android-sdk/sdk/native/jni
  if(ANDROID)
    if(WITH_OPENCV_STATIC)
      set(OpenCV_DIR ${OPENCV_DIRECTORY})
      find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
      include_directories(${OpenCV_INCLUDE_DIRS})
      # list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
      list(APPEND DEPEND_LIBS opencv_core opencv_video opencv_highgui opencv_imgproc opencv_imgcodecs)
    else()
      set(OpenCV_INCLUDE_DIRS ${OPENCV_DIRECTORY}/include)
      get_filename_component(OpenCV_NATIVE_DIR ${OPENCV_DIRECTORY} DIRECTORY)
      set(OpenCV_LIBS_DIR ${OpenCV_NATIVE_DIR}/libs)
      include_directories(${OpenCV_INCLUDE_DIRS})
      add_library(external_opencv_java STATIC IMPORTED GLOBAL)
      set_property(TARGET external_opencv_java PROPERTY IMPORTED_LOCATION 
        ${OpenCV_LIBS_DIR}/${ANDROID_ABI}/${OPENCV_ANDROID_SHARED_LIB_NAME})
      list(APPEND DEPEND_LIBS external_opencv_java)
    endif()
  # Win/Linux/Mac
  else()
    set(OpenCV_DIR ${OPENCV_DIRECTORY})
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
  endif()
else()
  message(STATUS "Use the default OpenCV lib from: ${OPENCV_URL}")
  if(ANDROID)
    if(WITH_OPENCV_STATIC)
      # When FastDeploy uses the OpenCV static library, there is no need to install OpenCV to FastDeploy thirds_libs
      download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install)
      if(EXISTS ${THIRD_PARTY_PATH}/install/opencv)
        file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/opencv) 
      endif()
      file(RENAME ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/ ${THIRD_PARTY_PATH}/install/opencv)
      set(OPENCV_FILENAME opencv)
      set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/sdk/native/jni)
      remove_redundant_opencv_android_abi_libraries(${OpenCV_DIR})
      find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR})
      include_directories(${OpenCV_INCLUDE_DIRS})
      # list(APPEND DEPEND_LIBS ${OpenCV_LIBS})
      list(APPEND DEPEND_LIBS opencv_core opencv_video opencv_highgui opencv_imgproc opencv_imgcodecs)
    else()
      # Installing OpenCV shared lib to FastDeploy third_libs/install dir.
      download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install)
      if(EXISTS ${THIRD_PARTY_PATH}/install/opencv)
        file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/opencv) 
      endif()
      file(RENAME ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/ ${THIRD_PARTY_PATH}/install/opencv)
      set(OPENCV_FILENAME opencv)
      set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/sdk/native/jni)
      remove_redundant_opencv_android_abi_libraries(${OpenCV_DIR})
      get_filename_component(OpenCV_NATIVE_DIR ${OpenCV_DIR} DIRECTORY)
      set(OpenCV_INCLUDE_DIRS ${OpenCV_DIR}/include)
      set(OpenCV_LIBS_DIR ${OpenCV_NATIVE_DIR}/libs)
      include_directories(${OpenCV_INCLUDE_DIRS})
      add_library(external_opencv_java STATIC IMPORTED GLOBAL)
      set_property(TARGET external_opencv_java PROPERTY IMPORTED_LOCATION 
        ${OpenCV_LIBS_DIR}/${ANDROID_ABI}/${OPENCV_ANDROID_SHARED_LIB_NAME})
      list(APPEND DEPEND_LIBS external_opencv_java)
    endif()
  # Win/Linux/Mac
  else()
    download_and_decompress(${OPENCV_URL} ${CMAKE_CURRENT_BINARY_DIR}/${OPENCV_FILENAME}${COMPRESSED_SUFFIX} ${THIRD_PARTY_PATH}/install/)
    if(EXISTS ${THIRD_PARTY_PATH}/install/opencv)
      file(REMOVE_RECURSE ${THIRD_PARTY_PATH}/install/opencv) 
    endif()
    file(RENAME ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME}/ ${THIRD_PARTY_PATH}/install/opencv)
    set(OPENCV_FILENAME opencv)
    if(NOT OpenCV_DIR)
      set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/${OPENCV_FILENAME})
    endif()
    if (WIN32)
      set(OpenCV_DIR ${OpenCV_DIR}/build)
    endif()
    find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)
    include_directories(${OpenCV_INCLUDE_DIRS})
    list(APPEND DEPEND_LIBS opencv_core opencv_video opencv_highgui opencv_imgproc opencv_imgcodecs)
  endif()
endif()
