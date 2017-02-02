#ifndef VCTRSKL_IMAGECOMPONENTLABELING_HPP
#define VCTRSKL_IMAGECOMPONENTLABELING_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"

namespace ImageComponentLabeling
{
    __device__ int findRootOfNodeLabel(int* labels, int current);
    __device__ bool areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV,
                                        const Color::byte& bY, const Color::byte& bU, const Color::byte& bV);
    __device__ void doUnionOfTrees(int* labels, int labelA, int labelB, int* didAnyLabelChange);
    __device__ void checkAndCombineTwoPixelRoots(int* labels, int labelA, int labelB, Color::byte* colorYUV,
                                                 int* didAnyLabelChange, int idxLimit);
    __global__ void createLocalComponentLabels(Color::byte* colorYUV, int* output, int width, int height);
    __global__ void mergeSolutionsOnBlockBorders(Color::byte* colorYUV, int* labels, int width, int height,
                                                 int tileSide);
    __global__ void flattenAllEquivalenceTrees(int* labels, int width, int height);

    void setComponentLabels(Color::byte* d_colorYUVData, int* d_componentLabels, unsigned int width,
                            unsigned int height);
}

#endif //VCTRSKL_IMAGECOMPONENTLABELING_HPP
