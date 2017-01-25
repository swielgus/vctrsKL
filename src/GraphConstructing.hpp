#ifndef VCTRSKL_COLOROPERATIONS_HPP
#define VCTRSKL_COLOROPERATIONS_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"
#include "PixelGraphInfo.hpp"

namespace GraphConstructing
{
    __device__ Color::byte getColorDifference(const Color::byte& a, const Color::byte& b);
    __device__ bool areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV,
                                        const Color::byte& bY, const Color::byte& bU, const Color::byte& bV);

    __device__ int getNeighborRowIdx(int row, GraphEdge direction);
    __device__ int getNeighborColIdx(int col, GraphEdge direction);

    __device__ Graph::byte getConnection(int row, int col, GraphEdge direction, const int* labelData, int width,
                                         int height);
    __global__ void
    createConnections(Graph::byte* edges, const int* labelData, int width, int height);
}

#endif //VCTRSKL_COLOROPERATIONS_HPP
