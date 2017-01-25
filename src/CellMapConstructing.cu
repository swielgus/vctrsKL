#include "CellMapConstructing.hpp"


__device__ bool CellMapConstructing::isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction)
{
    return nodeEdges & static_cast<Graph::byte>(direction);
}

__device__ Cell::byte CellMapConstructing::getTypeASideValueForGivenQuarter(const GraphEdge quarterDirection)
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
}

__global__ void
CellMapConstructing::createCells(Cell::byte* d_cellData, const Graph::byte* graphData, int width, int height)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height && col < width)
    {
        Cell::byte currentCellValue = 0;

        currentCellValue += CellMapConstructing::getQuarterOfCellSide(row, col, graphData, GraphEdge::UPPER_LEFT,
                                                                      width, height);
        currentCellValue += CellMapConstructing::getQuarterOfCellSide(row, col, graphData, GraphEdge::LOWER_LEFT,
                                                                      width, height);
        currentCellValue += CellMapConstructing::getQuarterOfCellSide(row, col, graphData, GraphEdge::LOWER_RIGHT,
                                                                      width, height);
        currentCellValue += CellMapConstructing::getQuarterOfCellSide(row, col, graphData, GraphEdge::UPPER_RIGHT,
                                                                      width, height);

        int idx = col + row * width;
        d_cellData[idx] = currentCellValue;
    }
}