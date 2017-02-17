#ifndef VCTRSKL_CURVECONTROLPOINTEXCLUDING_HPP
#define VCTRSKL_CURVECONTROLPOINTEXCLUDING_HPP

#include "PolygonSide.hpp"
#include "Constants.hpp"

namespace CurveControlPointExcluding
{
__constant__ __device__ CornerPatternEdge cornerPatterns[25*4];

__global__ void verifyWhichPointsAreToBeIgnored(PolygonSide* coordinateData, const PathPoint* pathData,
                                                bool* omitPoint, const unsigned int pathOffset, unsigned int width,
                                                unsigned int height, int pathLength, bool excludeTJunctions);
__device__ const Polygon::cord_type* getCoordinatesOfPathPoint(const int& pointIdx, const PathPoint* pathData,
                                                               PolygonSide* coordinateData, unsigned int width);
__device__ int getIdxOfRelativePoint(const int& source, int steps, const int& pathLength);
__device__ bool doesAPatternExist(const Polygon::cord_type* pointA, const Polygon::cord_type* pointB,
                                  const Polygon::cord_type* pointC, const Polygon::cord_type* pointD,
                                  const Polygon::cord_type* pointE, const CornerPatternEdge& step1,
                                  const CornerPatternEdge& step2, const CornerPatternEdge& step3,
                                  const CornerPatternEdge& step4);
__device__ bool doesTheEdgeFitThePattern(const Polygon::cord_type* pointA, const Polygon::cord_type* pointB,
                                         const CornerPatternEdge& expectedEdge);
__device__ CornerPatternEdge reversePatternEdge(const CornerPatternEdge& patternEdge);
__device__ int getRowDiffValueOfPattern(const CornerPatternEdge& patternEdge);
__device__ int getColDiffValueOfPattern(const CornerPatternEdge& patternEdge);
__device__ bool isPointOnIllegalImageBoundary(int idxOfPoint, const PathPoint* pathData,
                                              unsigned int width, unsigned int height);
__device__ bool doesPointHaveDegreeBiggerThanTwo(int idxOfPoint, const PathPoint& currentPathPoint,
                                                 PolygonSide* coordinateData, unsigned int width);

__host__ void allocateCornerPatternData();
}

#endif //VCTRSKL_CURVECONTROLPOINTEXCLUDING_HPP
