#ifndef VCTRSKL_POLYGONSIDEMAP_HPP
#define VCTRSKL_POLYGONSIDEMAP_HPP

#include "PixelGraph.hpp"
#include "PolygonSide.hpp"
#include "Constants.hpp"

class PolygonSideMap
{
public:
    PolygonSideMap() = delete;
    PolygonSideMap(const PixelGraph& graph);
    ~PolygonSideMap();

    std::vector<PolygonSide::Type> getInternalSideTypes();
private:
    const PixelGraph&              sourceGraph;
    std::vector<PolygonSide>       polygonSides;
    PolygonSide*                   d_polygonSides;

    void freeDeviceData();
    void constructInternalPolygonSides();
};


#endif //VCTRSKL_POLYGONSIDEMAP_HPP
