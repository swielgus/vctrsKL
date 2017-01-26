#include "CellMapConstructing.hpp"


__device__ bool CellMapConstructing::isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction)
{
    return nodeEdges & static_cast<Graph::byte>(direction);
}

/*__device__ Cell::byte CellMapConstructing::getTypeASideValueForGivenQuarter(const GraphEdge quarterDirection)
{
    CellSide result;
    switch(quarterDirection)
    {
        case GraphEdge::UPPER_RIGHT: result = CellSide::UPPER_RIGHT_TYPE_A; break;
        case GraphEdge::UPPER_LEFT:  result = CellSide::UPPER_LEFT_TYPE_A; break;
        case GraphEdge::LOWER_LEFT:  result = CellSide::LOWER_LEFT_TYPE_A; break;
        case GraphEdge::LOWER_RIGHT: result = CellSide::LOWER_RIGHT_TYPE_A; break;
        default: break;
    }
    return static_cast<Cell::byte>(result);
}
__device__ Cell::byte CellMapConstructing::getTypeBSideValueForGivenQuarter(const GraphEdge quarterDirection)
{
    CellSide result;
    switch(quarterDirection)
    {
        case GraphEdge::UPPER_RIGHT: result = CellSide::UPPER_RIGHT_TYPE_B; break;
        case GraphEdge::UPPER_LEFT:  result = CellSide::UPPER_LEFT_TYPE_B; break;
        case GraphEdge::LOWER_LEFT:  result = CellSide::LOWER_LEFT_TYPE_B; break;
        case GraphEdge::LOWER_RIGHT: result = CellSide::LOWER_RIGHT_TYPE_B; break;
        default: break;
    }
    return static_cast<Cell::byte>(result);
}
__device__ Cell::byte CellMapConstructing::getTypeCSideValueForGivenQuarter(const GraphEdge quarterDirection)
{
    CellSide result;
    switch(quarterDirection)
    {
        case GraphEdge::UPPER_RIGHT: result = CellSide::UPPER_RIGHT_TYPE_C; break;
        case GraphEdge::UPPER_LEFT:  result = CellSide::UPPER_LEFT_TYPE_C; break;
        case GraphEdge::LOWER_LEFT:  result = CellSide::LOWER_LEFT_TYPE_C; break;
        case GraphEdge::LOWER_RIGHT: result = CellSide::LOWER_RIGHT_TYPE_C; break;
        default: break;
    }
    return static_cast<Cell::byte>(result);
}

__device__ bool
CellMapConstructing::isThisSideTypeA(const Graph::byte nodeGraphData, const GraphEdge quarterDirection)
{
    return isThereAnEdge(nodeGraphData, quarterDirection);
}

__device__ GraphEdge CellMapConstructing::getTypeBEdgeToCheckInGivenQuarter(const GraphEdge quarterDirection)
{
    GraphEdge result;
    switch(quarterDirection)
    {
        case GraphEdge::UPPER_RIGHT: result = GraphEdge::UPPER_LEFT; break;
        case GraphEdge::UPPER_LEFT:  result = GraphEdge::UPPER_RIGHT; break;
        case GraphEdge::LOWER_LEFT:  result = GraphEdge::LOWER_RIGHT; break;
        case GraphEdge::LOWER_RIGHT: result = GraphEdge::LOWER_LEFT; break;
        default: break;
    }
    return result;
}

__device__ bool
CellMapConstructing::isThisSideTypeB(int row, int col, const Graph::byte* graphData, const GraphEdge quarterDirection,
                                     int width)
{
    bool result = false;
    int colOfCheckedNode = col - 1;
    if( quarterDirection == GraphEdge::UPPER_RIGHT || quarterDirection == GraphEdge::LOWER_RIGHT )
        colOfCheckedNode += 2;

    if(colOfCheckedNode >= 0 && colOfCheckedNode < width)
    {
        int idxOfCheckedNode = colOfCheckedNode + row * width;
        Graph::byte checkedGraphData = graphData[idxOfCheckedNode];
        GraphEdge edgeLookedFor = getTypeBEdgeToCheckInGivenQuarter(quarterDirection);

        result = isThereAnEdge(checkedGraphData, edgeLookedFor);
    }
    return result;
}

__device__ Cell::byte
CellMapConstructing::getQuarterOfCellSide(int row, int col, const Graph::byte* graphData,
                                          const GraphEdge quarterDirection, int width, int height)
{
    int idx = col + row * width;
    Graph::byte currentCellGraphData = graphData[idx];
    Cell::byte result;

    if( isThisSideTypeA(currentCellGraphData, quarterDirection) )
        result = getTypeASideValueForGivenQuarter(quarterDirection);
    else if( isThisSideTypeB(row, col, graphData, quarterDirection, width) )
        result = getTypeBSideValueForGivenQuarter(quarterDirection);
    else
        result = getTypeCSideValueForGivenQuarter(quarterDirection);

    return result;
}*/

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

        cellData[idxOfQuarter].type = currentQuarterSideType;
        cellData[idxOfQuarter].pointA[0] = rowA;
        cellData[idxOfQuarter].pointA[1] = colA;
        cellData[idxOfQuarter].pointB[0] = rowB;
        cellData[idxOfQuarter].pointB[1] = colB;
    }
}