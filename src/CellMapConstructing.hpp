#ifndef VCTRSKL_CELLMAPCONSTRUCTING_HPP
#define VCTRSKL_CELLMAPCONSTRUCTING_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"

namespace CellMapConstructing
{
    __device__ bool isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction);
    __device__ bool isThisQuarterForwardSlash(int row, int col, int graphWidth, const Graph::byte* graphData);
    __device__ bool isThisQuarterBackslash(int row, int col, int graphWidth, const Graph::byte* graphData);
    __global__ void createCells(CellSide* cellData, const Graph::byte* graphData, int width, int height);
}

#endif //VCTRSKL_CELLMAPCONSTRUCTING_HPP
