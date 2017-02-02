#include "PolygonSideMapConstructing.hpp"
#include "PolygonSideMap.hpp"

PolygonSideMap::PolygonSideMap(const PixelGraph& graph)
        : sourceGraph{graph}, polygonSides{}, d_polygonSides{nullptr}, regionConstructor{nullptr}
{
    constructInternalPolygonSides();
    generateRegionBoundaries();
    allocatePathPointsOfBoundariesOnDevice();
}

PolygonSideMap::~PolygonSideMap()
{
    freeDeviceData();
    delete regionConstructor;
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

const std::vector<PolygonSide>& PolygonSideMap::getInternalSides() const
{
    return polygonSides;
}

void PolygonSideMap::generateRegionBoundaries()
{
    const std::size_t width = sourceGraph.getWidth();
    const std::size_t height = sourceGraph.getHeight();
    regionConstructor = new RegionConstructor{getInternalSides(), sourceGraph.get1DEdgeValues(), width, height};
}

const ClipperLib::Paths& PolygonSideMap::getGeneratedRegionBoundaries() const
{
    return regionConstructor->getBoundaries();
}

void PolygonSideMap::allocatePathPointsOfBoundariesOnDevice()
{
    auto pathPoints = regionConstructor->createPathPoints();
    //TODO send this vector of vectors to device
}

std::vector<std::vector<PathPoint> > PolygonSideMap::getPathPointBoundaries() const
{
    return regionConstructor->createPathPoints();
}
