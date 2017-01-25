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
GraphConstructing::getConnection(int row, int col, GraphEdge direction, const int* labelData, int width, int height)
{
    std::size_t idx = col + row * width;

    int neighborRow = GraphConstructing::getNeighborRowIdx(row, direction);
    int neighborCol = GraphConstructing::getNeighborColIdx(col, direction);

    Graph::byte result = 0;
    if( (neighborRow >= 0 && neighborRow < height) && (neighborCol >= 0 && neighborCol < width) )
    {
        std::size_t comparedIdx = neighborCol + neighborRow * width;

        if( labelData[idx] == labelData[comparedIdx] )
        {
            bool isThisEdgeRedundant = false;
            if( (neighborRow != row) && (neighborCol != col) )
            {
                int secondNeighborIdx = col + neighborRow * width;
                int thirdNeighborIdx = neighborCol + row * width;
                isThisEdgeRedundant = labelData[secondNeighborIdx] == labelData[thirdNeighborIdx] &&
                                      labelData[secondNeighborIdx] == labelData[idx];
            }

            if(!isThisEdgeRedundant)
                result = static_cast<Graph::byte>(direction);
        }
    }
    return result;
}

__global__ void
GraphConstructing::createConnections(Graph::byte* edges, const int* labelData, int width, int height)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height && col < width)
    {
        int idx = col + row * width;
        Graph::byte currentNodeEdgeValues = 0;

        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::UP, labelData, width, height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::DOWN, labelData, width, height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::LEFT, labelData, width, height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::RIGHT, labelData, width, height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::UPPER_RIGHT, labelData, width,
                                                                  height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::UPPER_LEFT, labelData, width,
                                                                  height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::LOWER_RIGHT, labelData, width,
                                                                  height);
        currentNodeEdgeValues += GraphConstructing::getConnection(row, col, GraphEdge::LOWER_LEFT, labelData, width,
                                                                  height);

        edges[idx] = currentNodeEdgeValues;
    }
}