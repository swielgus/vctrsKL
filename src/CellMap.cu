#include "CellMap.hpp"
#include "CellMapConstructing.hpp"

CellMap::CellMap(const PixelGraph& graph)
    : sourceGraph{graph}, d_cellData{nullptr}
{
    constructPixelCells();
}

CellMap::~CellMap()
{
    freeDeviceData();
}

void CellMap::freeDeviceData()
{
    cudaFree(d_cellData);
}

std::vector< std::vector<CellSideType> > CellMap::getCellTypes() const
{
    const std::size_t width = sourceGraph.getWidth() + 1;
    const std::size_t height = sourceGraph.getHeight() + 1;

    std::vector< std::vector<CellSideType> > result;
    result.resize(height);
    for(std::vector<cell_type>& row : result)
        row.resize(width);

    CellSide* cellSideValues = new CellSide[width * height];
    cudaMemcpy(cellSideValues, d_cellData, width * height * sizeof(CellSide), cudaMemcpyDeviceToHost);

    Cell::byte firstThreeBitsMask = 7;
    for(std::size_t x = 0; x < height; ++x)
        for(std::size_t y = 0; y < width; ++y)
        {
            result[x][y] = static_cast<CellSideType>(((cellSideValues + (y + x * width))->type) & firstThreeBitsMask);
        }

    delete[] cellSideValues;

    return result;
}

void CellMap::constructPixelCells()
{
    const std::size_t width = sourceGraph.getWidth() + 1;
    const std::size_t height = sourceGraph.getHeight() + 1;
    cudaMalloc( &d_cellData, width * height * sizeof(CellSide));

    const PixelGraph::edge_type* d_graphData = sourceGraph.getGPUAddressOfGraphData();

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    CellMapConstructing::createCells<<<dimGrid, dimBlock>>>(d_cellData, d_graphData, width, height);
    cudaDeviceSynchronize();
}