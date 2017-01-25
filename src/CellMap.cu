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

std::vector< std::vector<CellMap::cell_type> > CellMap::getCellValues() const
{
    const std::size_t width = sourceGraph.getWidth();
    const std::size_t height = sourceGraph.getHeight();

    std::vector< std::vector<cell_type> > result;
    result.resize(height);
    for(std::vector<cell_type>& row : result)
        row.resize(width);

    cell_type* cellSideValues = new cell_type[width * height];
    cudaMemcpy(cellSideValues, d_cellData, width * height * sizeof(cell_type), cudaMemcpyDeviceToHost);

    for(std::size_t x = 0; x < height; ++x)
        for(std::size_t y = 0; y < width; ++y)
        {
            result[x][y] = *(cellSideValues + (y + x * width));
        }

    delete[] cellSideValues;

    return result;
}

void CellMap::constructPixelCells()
{
    const std::size_t width = sourceGraph.getWidth();
    const std::size_t height = sourceGraph.getHeight();
    cudaMalloc( &d_cellData, width * height * sizeof(cell_type));

    const PixelGraph::edge_type* d_graphData = sourceGraph.getGPUAddressOfGraphData();

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    CellMapConstructing::createCells<<<dimGrid, dimBlock>>>(d_cellData, d_graphData, width, height);
    cudaDeviceSynchronize();
}