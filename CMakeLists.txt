cmake_minimum_required(VERSION 3.5)
project(vctrsKL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -Wall -pedantic -Wextra")

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-arch=sm_35;-std=c++11;-O2;-DVERBOSE")
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} \"-D_FORCE_INLINES\" ")

add_subdirectory(src)
add_subdirectory(test)
