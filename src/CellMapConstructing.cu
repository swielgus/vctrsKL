#include "CellMapConstructing.hpp"


__device__ bool CellMapConstructing::isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction)
{
    return nodeEdges & static_cast<Graph::byte>(direction);
}

__device__ bool
CellMapConstructing::isThisQuarterForwardSlash(int row, int col, int graphWidth, const Graph::byte* graphData)
{
    int idxOfGraphEntry = col + row * graphWidth;
    Graph::byte checkedGraphData = graphData[idxOfGraphEntry];
    return isThereAnEdge(checkedGraphData, GraphEdge::UPPER_LEFT);
}

__device__ bool
CellMapConstructing::isThisQuarterBackslash(int row, int col, int graphWidth, const Graph::byte* graphData)
{
    int idxOfGraphEntry = col - 1 + row * graphWidth;
    Graph::byte checkedGraphData = graphData[idxOfGraphEntry];
    return isThereAnEdge(checkedGraphData, GraphEdge::UPPER_RIGHT);
}

__global__ void
CellMapConstructing::createCells(CellSide* cellData, const Graph::byte* graphData, int width, int height)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height && col < width)
    {
        int idxOfQuarter = col + row * width;
        CellSideType currentQuarterSideType = CellSideType::Point;
        if(row != 0 && col != 0 && row != height - 1 && col != width - 1)
        {
            if(isThisQuarterForwardSlash(row, col, width-1, graphData))
                currentQuarterSideType = CellSideType::ForwardSlash;
            else if(isThisQuarterBackslash(row, col, width-1, graphData))
                currentQuarterSideType = CellSideType::Backslash;
        }

        Cell::cord_type rowA = 0.0;
        Cell::cord_type colA = 0.0;
        Cell::cord_type rowB = 0.0;
        Cell::cord_type colB = 0.0;
        if(currentQuarterSideType == CellSideType::ForwardSlash)
        {
            rowA = static_cast<Cell::cord_type>(row) - 0.75f;
            colA = static_cast<Cell::cord_type>(col) - 0.25f;
            rowB = static_cast<Cell::cord_type>(row) - 0.25f;
            colB = static_cast<Cell::cord_type>(col) - 0.75f;
        }
        else if(currentQuarterSideType == CellSideType::Backslash)
        {
            rowA = static_cast<Cell::cord_type>(row) - 0.75f;
            colA = static_cast<Cell::cord_type>(col) - 0.75f;
            rowB = static_cast<Cell::cord_type>(row) - 0.25f;
            colB = static_cast<Cell::cord_type>(col) - 0.25f;
        }

        cellData[idxOfQuarter].type = static_cast<Cell::byte>(currentQuarterSideType);
        cellData[idxOfQuarter].pointA[0] = rowA;
        cellData[idxOfQuarter].pointA[1] = colA;
        cellData[idxOfQuarter].pointB[0] = rowB;
        cellData[idxOfQuarter].pointB[1] = colB;
    }
}