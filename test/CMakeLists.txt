add_subdirectory(googletest/googlemock/ gtest)
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/src/)

add_executable(ImageDataTest ImageDataTest.cpp)
target_link_libraries(ImageDataTest ImageData gtest gtest_main)
