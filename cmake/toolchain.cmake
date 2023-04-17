if (DEFINED TARGET_ABI)
    set(CMAKE_SYSTEM_NAME Linux)
    set(CMAKE_BUILD_TYPE MinSizeRel)
    if(${TARGET_ABI} MATCHES "armhf")
        set(CMAKE_SYSTEM_PROCESSOR arm)
        if(NOT CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER "arm-linux-gnueabihf-gcc")
        endif()
        if(NOT CMAKE_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "arm-linux-gnueabihf-g++")
        endif()
        set(CMAKE_CXX_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv7-a -mfloat-abi=hard -mfpu=neon-vfpv4 ${CMAKE_C_FLAGS}" )
        set(OPENCV_URL "https://bj.bcebos.com/fastdeploy/third_libs/opencv-linux-armv7hf-4.6.0.tgz")
        set(OPENCV_FILENAME "opencv-linux-armv7hf-4.6.0")
        if(WITH_TIMVX)
            set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/third_libs/lite-linux-armhf-timvx-20230316.tgz")
        else()
            message(STATUS "PADDLELITE_URL will be configured if WITH_TIMVX=ON.")
        endif()
        set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)
        set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/opencv/lib/cmake/opencv4)
    elseif(${TARGET_ABI} MATCHES "arm64")
        set(CMAKE_SYSTEM_PROCESSOR aarch64)
        if(NOT CMAKE_C_COMPILER)
            set(CMAKE_C_COMPILER "aarch64-linux-gnu-gcc")
        endif()
        if(NOT CMAKE_CXX_COMPILER)
            set(CMAKE_CXX_COMPILER "aarch64-linux-gnu-g++")
        endif()
        set(CMAKE_CXX_FLAGS "-march=armv8-a ${CMAKE_CXX_FLAGS}")
        set(CMAKE_C_FLAGS "-march=armv8-a ${CMAKE_C_FLAGS}")
        set(OPENCV_URL "https://bj.bcebos.com/fastdeploy/third_libs/opencv-linux-aarch64-4.6.0.tgz")
        set(OPENCV_FILENAME "opencv-linux-aarch64-4.6.0")
        if(WITH_TIMVX)
            set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/third_libs/lite-linux-aarch64-timvx-20230316.tgz")
        else()
            set(PADDLELITE_URL "https://bj.bcebos.com/fastdeploy/third_libs/lite-linux-arm64-20230316.tgz")
        endif()
        set(THIRD_PARTY_PATH ${CMAKE_CURRENT_BINARY_DIR}/third_libs)
        set(OpenCV_DIR ${THIRD_PARTY_PATH}/install/opencv/lib/cmake/opencv4)
    else()
        message(FATAL_ERROR "When cross-compiling, please set the -DTARGET_ABI to arm64 or armhf.")
    endif()
endif()

