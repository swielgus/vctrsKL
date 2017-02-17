#ifndef VCTRSKL_GRAPHCROSSRESOLVING_HPP
#define VCTRSKL_GRAPHCROSSRESOLVING_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"

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
    __device__ int  getLengthOfPathComponent(int row, int col, GraphEdge secondaryNodeDirection, Graph::byte* edges,
                                             int width, int height);
    __device__ int getSizeOfConnectedComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                               const int radius, const int* labelData, int width, int height);

    __global__ void resolveCriticalCrossings(Graph::byte* edges, const int* labelData, int width, int height,
                                             const int islandHeuristicMultiplier, const int curveHeuristicMultiplier,
                                             const int sparsePixelsMultiplier, const int sparsePixelsRadius);
}

#endif //VCTRSKL_GRAPHCROSSRESOLVING_HPP
