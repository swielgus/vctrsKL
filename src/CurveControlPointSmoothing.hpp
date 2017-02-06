#ifndef VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP
#define VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP

#include "PolygonSide.hpp"
#include "CurveCurvature.hpp"

namespace CurveControlPointSmoothing
{
__global__ void optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData, const unsigned int pathOffset,
                              unsigned int width, unsigned int height, Polygon::cord_type* randomShiftPointValues,
                              const Polygon::cord_type radius, bool* omitPoint, int pathLength);
__device__ Polygon::cord_type getCoordinateOfPathPoint(const int& pointIdx, const PathPoint* pathData,
                                                       PolygonSide* coordinateData, unsigned int width,
                                                       unsigned int height, int coordIdx);
__device__ Polygon::cord_type getPositionalEnergy(const ControlPoint& pointA, const ControlPoint& pointB);
__device__ Polygon::cord_type getCurvatureIntegral(const ControlPoint& startPoint, const ControlPoint& midPoint,
                                                   const ControlPoint& endPoint, const int numberOfSamples);
__device__ int getIdxOfRelativePoint(const int& source, int steps, const int& pathLength);
__device__ void setNewCoordinatesOfPathPoint(const int& pointIdx, const PathPoint* pathData,
                                             PolygonSide* coordinateData, unsigned int width,
                                             unsigned int height, Polygon::cord_type resultRow,
                                             Polygon::cord_type resultCol);
__device__ ControlPoint getRandomPointInNeighborhood(const ControlPoint& source, const Polygon::cord_type& radius,
                                                     Polygon::cord_type* randomShiftPointValues);
}

#endif //VCTRSKL_CURVECONTROLPOINTSMOOTHING_HPP
