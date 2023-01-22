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

if(NOT WITH_OPENMP)
 message(FATAL_ERROR "Please set WITH_OPENMP=ON before inclue openmp.cmake")
endif()

if(ANDROID)
  if(ARM_TARGET_LANG STREQUAL "gcc")
    set(OpenMP_C_FLAGS "-fopenmp")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY omp)
    set(OpenMP_C_FLAGS_WORK "-fopenmp")
    set(OpenMP_C_LIB_NAMES_WORK "omp")
    set(OpenMP_CXX_FLAGS_WORK "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES_WORK "omp")
  endif()

  find_package(OpenMP REQUIRED)
  if(WITH_LITE_STATIC)
    # Can use static/shared omp lib if WITH_LITE_STATIC=ON
    set(LIBOMP_ENABLE_SHARED OFF)
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
      add_definitions(-DWITH_OPENMP)
      if(${ANDROID_NDK_MAJOR})
        if(${ANDROID_NDK_MAJOR} GREATER 20)
          message(STATUS "ANDROID_NDK_MAJOR GREATER 20")
          set(OPENMP_LINK_FLAGS "-fopenmp -static-openmp")
        else()
          set(OPENMP_LINK_FLAGS "-fopenmp")
        endif()  
      endif()
    else()
      message(FATAL_ERROR "Could not found OpenMP!")  
    endif()
  else()
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND)
      add_definitions(-DWITH_OPENMP)
      # Can only use shared omp lib if WITH_LITE_STATIC=OFF
      set(OPENMP_LINK_FLAGS "-fopenmp")
    else()
      message(FATAL_ERROR "Could not found OpenMP!")  
    endif()  
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OPENMP_LINK_FLAGS}")
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${OPENMP_LINK_FLAGS}")
  message(STATUS "Found OpenMP ${OpenMP_VERSION} ${OpenMP_CXX_VERSION}")
  message(STATUS "OpenMP C flags:  ${OpenMP_C_FLAGS}")
  message(STATUS "OpenMP CXX flags:  ${OpenMP_CXX_FLAGS}")
  message(STATUS "OpenMP EXE LINKER flags:  ${OPENMP_LINK_FLAGS}")
  message(STATUS "OpenMP OpenMP_CXX_LIB_NAMES:  ${OpenMP_CXX_LIB_NAMES}")
  message(STATUS "OpenMP OpenMP_CXX_LIBRARIES:  ${OpenMP_CXX_LIBRARIES}")

elseif(WIN32)
  message(FATAL_ERROR "WITH_OPENMP=ON option is not support for WIN32 now!")
elseif(APPLE)
  message(FATAL_ERROR "WITH_OPENMP=ON option is not support for APPLE now!")
elseif(IOS)
  message(FATAL_ERROR "WITH_OPENMP=ON option is not support for IOS now!")  
else()
  message(FATAL_ERROR "WITH_OPENMP=ON option is not support for Linux now!")
endif()
