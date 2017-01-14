cmake_minimum_required(VERSION 3.5)
project(vctrsKL)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -Wall -pedantic -Wextra")

find_package(CUDA REQUIRED)

add_subdirectory(src)
add_subdirectory(test)
