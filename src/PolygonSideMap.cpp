#include "PolygonSideMapConstructing.hpp"
#include "PolygonSideMap.hpp"

PolygonSideMap::PolygonSideMap(const PixelGraph& graph)
        : sourceGraph{graph}, polygonSides{}, d_polygonSides{nullptr}
{
    constructInternalPolygonSides();
}

PolygonSideMap::~PolygonSideMap()
{
    freeDeviceData();
}

void PolygonSideMap::freeDeviceData()
{
    cudaFree(d_polygonSides);
}

std::vector<PolygonSide::Type> PolygonSideMap::getInternalSideTypes()
{
    const std::size_t width = sourceGraph.getWidth();
    const std::size_t height = sourceGraph.getHeight();

    std::vector<PolygonSide::Type> result;
    result.resize(height * width);

    for(std::size_t idx = 0; idx < height * width; ++idx)
        result[idx] = polygonSides[idx].getType();

    return result;
}

void PolygonSideMap::constructInternalPolygonSides()
{
    const std::size_t width = sourceGraph.getWidth();
    const std::size_t height = sourceGraph.getHeight();
    cudaMalloc( &d_polygonSides, width * height * sizeof(PolygonSide));

    const PixelGraph::edge_type* d_graphData = sourceGraph.getGPUAddressOfGraphData();

    PolygonSideMapConstructing::createMap(d_polygonSides, d_graphData, width, height);

    PolygonSideMapConstructing::getCreatedMapData(polygonSides, d_polygonSides, width, height);
}