cmake_minimum_required(VERSION 3.5)
project(vctrsKL)

INCLUDE(FindCUDA)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -Wextra")

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-G;-g;-arch=sm_35;-O2;-DVERBOSE;-D_FORCE_INLINES")
set(CUDA_PROPAGATE_HOST_FLAGS ON)
set(CUDA_SEPARABLE_COMPILATION OFF)
set(CUDA_VERBOSE_BUILD OFF)

include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
add_subdirectory(test)
