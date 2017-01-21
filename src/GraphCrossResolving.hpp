#ifndef VCTRSKL_GRAPHCROSSRESOLVING_HPP
#define VCTRSKL_GRAPHCROSSRESOLVING_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"
#include "PixelGraphInfo.hpp"

namespace GraphCrossResolving
{
    __device__ void doAtomicAnd(Graph::byte* address, Graph::byte value);
    __device__ void doAtomicOr(Graph::byte* address, Graph::byte value);
    __device__ void addEdgeConnection(Graph::byte& nodeEdges, GraphEdge direction);
    __device__ void removeEdgeConnection(Graph::byte& nodeEdges, GraphEdge direction);
    __device__ bool isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction);
    __device__ int  getNodeDegree(const Graph::byte& nodeEdges);
    __device__ bool isThereAnIslandNode(const Graph::byte& nodeAEdges, const Graph::byte& nodeBEdges);
    __device__ int  getNeighborRowIdx(int row, GraphEdge direction);
    __device__ int  getNeighborColIdx(int col, GraphEdge direction);
    __device__ GraphEdge getNeighborInDirectionOtherThanGiven(const Graph::byte& nodeEdges,
                                                              const GraphEdge forbiddenDirection);
    __device__ GraphEdge getOppositeDirection(GraphEdge direction);
    __device__ int  getLengthOfPathComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                             PixelGraphInfo* graphInfo);
    __device__ int getSizeOfConnectedComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                               const std::size_t& radius, PixelGraphInfo* graphInfo);

    __global__ void removeUnnecessaryCrossings(PixelGraphInfo* graphInfo);
    __global__ void resolveCriticalCrossings(PixelGraphInfo* graphInfo);
}

#endif //VCTRSKL_GRAPHCROSSRESOLVING_HPP
