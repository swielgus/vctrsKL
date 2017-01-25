#ifndef VCTRSKL_CELLMAPCONSTRUCTING_HPP
#define VCTRSKL_CELLMAPCONSTRUCTING_HPP

#include "Constants.hpp"

namespace CellMapConstructing
{
    __device__ bool isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction);
    __device__ Cell::byte getTypeASideValueForGivenQuarter(const GraphEdge quarterDirection);
    __device__ Cell::byte getTypeBSideValueForGivenQuarter(const GraphEdge quarterDirection);
    __device__ Cell::byte getTypeCSideValueForGivenQuarter(const GraphEdge quarterDirection);
    __device__ GraphEdge getTypeBEdgeToCheckInGivenQuarter(const GraphEdge quarterDirection);
    __device__ bool isThisSideTypeA(const Graph::byte graphData, const GraphEdge quarterDirection);
    __device__ bool isThisSideTypeB(int row, int col, const Graph::byte* graphData, const GraphEdge quarterDirection,
                                    int width);
    __device__ Cell::byte getQuarterOfCellSide(int row, int col, const Graph::byte* graphData,
                                               const GraphEdge quarterDirection, int width, int height);
    __global__ void createCells(Cell::byte* d_cellData, const Graph::byte* graphData, int width, int height);
}

#endif //VCTRSKL_CELLMAPCONSTRUCTING_HPP
