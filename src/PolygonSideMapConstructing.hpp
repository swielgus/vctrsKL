#ifndef VCTRSKL_CELLMAPCONSTRUCTING_HPP
#define VCTRSKL_CELLMAPCONSTRUCTING_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector>
#include "PolygonSide.hpp"
#include "Constants.hpp"

namespace PolygonSideMapConstructing
{
    __device__ bool isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction);
    __device__ bool isThisQuarterForwardSlash(int row, int col, int width, const Graph::byte* graphData);
    __device__ bool isThisQuarterBackslash(int row, int col, int width, const Graph::byte* graphData);
    __global__ void createPolygonSide(PolygonSide* sideData, const Graph::byte* graphData, int width, int height);
    void createMap(PolygonSide* sideData, const Graph::byte* graphData, int width, int height);
    void getCreatedMapData(std::vector<PolygonSide>& output, PolygonSide* d_sideData, int width, int height);
}

#endif //VCTRSKL_CELLMAPCONSTRUCTING_HPP
