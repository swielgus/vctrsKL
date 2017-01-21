#include <cstdlib>
#include "GraphConstructing.hpp"

__device__ Color::byte GraphConstructing::getColorDifference(const Color::byte& a, const Color::byte& b)
{
    return (a < b) ? (b-a) : (a-b);
}

__device__ bool
GraphConstructing::areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV,
                                       const Color::byte& bY, const Color::byte& bU, const Color::byte& bV)
{
    const Color::byte thresholdY = 48;
    const Color::byte thresholdU = 7;
    const Color::byte thresholdV = 6;

    //return (getColorDifference(aY, bY) <= thresholdY) && (getColorDifference(aU, bU) <= thresholdU) &&
    //       (getColorDifference(aV, bV) <= thresholdV);
    return abs(aY - bY) <= thresholdY && abs(aU - bU) <= thresholdU && abs(aV - bV) <= thresholdV;
}

__device__
int GraphConstructing::getNeighborRowIdx(int row, GraphEdge direction)
{
    if(direction == GraphEdge::UP || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::UPPER_RIGHT)
        row--;
    if(direction == GraphEdge::DOWN || direction == GraphEdge::LOWER_LEFT || direction == GraphEdge::LOWER_RIGHT)
        row++;

    return row;
}

__device__
int GraphConstructing::getNeighborColIdx(int col, GraphEdge direction)
{
    if(direction == GraphEdge::LEFT || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::LOWER_LEFT)
        col--;
    if(direction == GraphEdge::RIGHT || direction == GraphEdge::UPPER_RIGHT || direction == GraphEdge::LOWER_RIGHT)
        col++;

    return col;
}

__device__ Graph::byte
GraphConstructing::getConnection(int row, int col, GraphEdge direction, PixelGraphInfo* graphInfo,
                                 const Color::byte* colorY, const Color::byte* colorU, const Color::byte* colorV)
{
    std::size_t idx = col + row * graphInfo->width;

    int neighborRow = GraphConstructing::getNeighborRowIdx(row, direction);
    int neighborCol = GraphConstructing::getNeighborColIdx(col, direction);

    Graph::byte result = 0;
    if( (neighborRow >= 0 && neighborRow < graphInfo->height) && (neighborCol >= 0 && neighborCol < graphInfo->width) )
    {
        std::size_t comparedIdx = neighborCol + neighborRow * graphInfo->width;
        if( GraphConstructing::areYUVColorsSimilar(colorY[idx], colorU[idx], colorV[idx],
                                                   colorY[comparedIdx], colorU[comparedIdx], colorV[comparedIdx]))
            result = static_cast<Graph::byte>(direction);
    }
    return result;
}

__global__ void
GraphConstructing::createConnections(PixelGraphInfo* graphInfo, const Color::byte* colorY,
                                     const Color::byte* colorU, const Color::byte* colorV)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    if(i < graphInfo->height && j < graphInfo->width)
    {
        std::size_t idx = j + i * graphInfo->width;
        graphInfo->edges[idx] = 0;

        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::UP, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::DOWN, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::LEFT, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::RIGHT, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::UPPER_RIGHT, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::UPPER_LEFT, graphInfo,
                                                                  colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::LOWER_RIGHT, graphInfo, colorY, colorU, colorV);
        graphInfo->edges[idx] += GraphConstructing::getConnection(i, j, GraphEdge::LOWER_LEFT, graphInfo, colorY, colorU, colorV);
    }
}