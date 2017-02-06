#ifndef VCTRSKL_CURVEOPTIMIZER_HPP
#define VCTRSKL_CURVEOPTIMIZER_HPP

#include "PolygonSideMap.hpp"
#include "Constants.hpp"

class CurveOptimizer
{
public:
    CurveOptimizer() = delete;
    CurveOptimizer(PolygonSideMap& usedSideMap);
    ~CurveOptimizer();
private:
    unsigned int                         imageWidth;
    unsigned int                         imageHeight;
    PolygonSide*                         d_coordinateData;
    std::vector<std::vector<PathPoint> > usedPathPoints;
    std::vector<unsigned int>            pathAddressOffsets;
    PathPoint*                           d_pathPointData;
    bool*                                d_omitPointDuringOptimization;

    void allocatePathPointDataOnDevice();
    void optimizeEnergyInAllPaths();
    void checkWhichPointsAreToBeOmitted();
};


#endif //VCTRSKL_CURVEOPTIMIZER_HPP
