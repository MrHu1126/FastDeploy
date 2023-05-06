# Check for 32bit system
if(WIN32)
  if(NOT CMAKE_CL_64)
    message("***********************Compile on non 64-bit system now**********************")
    add_definitions(-DNON_64_PLATFORM)
    if(WITH_GPU)
      message(FATAL_ERROR "-DWITH_GPU=ON doesn't support on non 64-bit system now.")
    endif()
    if(ENABLE_PADDLE_BACKEND)
      message(FATAL_ERROR "-DENABLE_PADDLE_BACKEND=ON doesn't support on non 64-bit system now.")
    endif()
    if(ENABLE_POROS_BACKEND)
      message(FATAL_ERROR "-DENABLE_POROS_BACKEND=ON doesn't support on non 64-bit system now.")
    endif()
  endif()
endif()

if(ANDROID OR IOS)
  if(ENABLE_ORT_BACKEND)
    message(FATAL_ERROR "Not support ONNXRuntime backend for Andorid/IOS now. Please set ENABLE_ORT_BACKEND=OFF.")
  endif()
  if(ENABLE_PADDLE_BACKEND)
    message(FATAL_ERROR "Not support Paddle backend for Andorid/IOS now. Please set ENABLE_PADDLE_BACKEND=OFF.")
  endif()
  if(ENABLE_OPENVINO_BACKEND)
    message(FATAL_ERROR "Not support OpenVINO backend for Andorid/IOS now. Please set ENABLE_OPENVINO_BACKEND=OFF.")
  endif()
  if(ENABLE_TRT_BACKEND)
    message(FATAL_ERROR "Not support TensorRT backend for Andorid/IOS now. Please set ENABLE_TRT_BACKEND=OFF.")
  endif()
endif()

if(WITH_GPU)
  if(APPLE)
    message(FATAL_ERROR "Cannot enable GPU while compling in Mac OSX.")
    set(WITH_GPU OFF)
  elseif(ANDROID OR IOS)
    message(FATAL_ERROR "Cannot enable GPU while compling in Android or IOS.")
    set(WITH_GPU OFF)
  endif()
endif()

if(WITH_OPENCL)
  if(NOT ANDROID OR NOT ENABLE_LITE_BACKEND)
    message(FATAL_ERROR "Cannot enable OpenCL while compling unless in Android and Paddle Lite backend is enbaled.")
    set(WITH_GPU OFF)
  endif()
endif()
