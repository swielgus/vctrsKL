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
    const ClipperLib::Paths& getGeneratedRegionBoundaries() const;
    std::vector< std::vector<PathPoint> > getPathPointBoundaries() const;
    const std::vector<ClipperLib::IntPoint>& getColorRepresentatives() const;
    PolygonSide* getGPUAddressOfPolygonCoordinateData();
    unsigned int getImageWidth() const;
    unsigned int getImageHeight() const;
    std::vector<PolygonSide> getInternalSidesFromDevice() const;
private:
    const PixelGraph&              sourceGraph;
    std::vector<PolygonSide>       polygonSides;
    PolygonSide*                   d_polygonSides;
    RegionConstructor*             regionConstructor;

    void freeDeviceData();
    void constructInternalPolygonSides();
    void generateRegionBoundaries();
    const std::vector<PolygonSide>& getInternalSides() const;
};


#endif //VCTRSKL_POLYGONSIDEMAP_HPP
