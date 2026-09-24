cmake_minimum_required(VERSION 3.10)
project(litemono_trt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDA REQUIRED)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/library not found")
endif()

add_library(litemono_trt SHARED litemono_trt.cpp)
target_include_directories(litemono_trt PRIVATE
  ${TENSORRT_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS})
target_link_libraries(litemono_trt PRIVATE
  ${TENSORRT_LIBRARY}
  ${CUDA_CUDART_LIBRARY})
