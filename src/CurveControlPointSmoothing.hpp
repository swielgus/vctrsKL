#ifndef VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP
#define VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP

#include "PolygonSide.hpp"

namespace CurveControlPointSmoothing
{
__global__ void optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData, const unsigned int pathOffset,
                              unsigned int width, unsigned int height);
__device__ PolygonSide::point_type getCoordinateOfPathPoint(const int& pointIdx, const PathPoint* pathData,
                                                            PolygonSide* coordinateData, unsigned int width,
                                                            unsigned int height, int coordIdx);
}

#endif //VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP
