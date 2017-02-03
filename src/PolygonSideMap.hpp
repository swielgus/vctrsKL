#ifndef VCTRSKL_POLYGONSIDEMAP_HPP
#define VCTRSKL_POLYGONSIDEMAP_HPP

#include "PixelGraph.hpp"
#include "PolygonSide.hpp"
#include "RegionConstructor.hpp"
#include "Constants.hpp"

class PolygonSideMap
{
public:
    PolygonSideMap() = delete;
    PolygonSideMap(const PixelGraph& graph);
    ~PolygonSideMap();

    std::vector<PolygonSide::Type> getInternalSideTypes();
    const std::vector<PolygonSide>& getInternalSides() const;
    const ClipperLib::Paths& getGeneratedRegionBoundaries() const;
    std::vector< std::vector<PathPoint> > getPathPointBoundaries() const;
    const std::vector<ClipperLib::IntPoint>& getColorRepresentatives() const;
private:
    const PixelGraph&              sourceGraph;
    std::vector<PolygonSide>       polygonSides;
    PolygonSide*                   d_polygonSides;
    RegionConstructor*             regionConstructor;

    void freeDeviceData();
    void constructInternalPolygonSides();
    void generateRegionBoundaries();
    void allocatePathPointsOfBoundariesOnDevice();
};


#endif //VCTRSKL_POLYGONSIDEMAP_HPP
