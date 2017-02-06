#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveControlPointExcluding.hpp"

__global__ void CurveControlPointExcluding::verifyWhichPointsAreToBeIgnored(PolygonSide* coordinateData,
                                                                            const PathPoint* pathData,
                                                                            bool* omitPoint,
                                                                            const unsigned int pathOffset,
                                                                            unsigned int width, unsigned int height,
                                                                            int pathLength)
{
    int idxOfPoint = blockIdx.x * blockDim.x + threadIdx.x;
    if(idxOfPoint >= pathLength)
    {
        return;
    }
    __syncthreads();

    const PathPoint* currentPathData = pathData + pathOffset;
    bool* currentPathOmittingData = omitPoint + pathOffset;

    if( blockDim.x < pathLength )
    {
        if( threadIdx.x < 5 || threadIdx.x >= blockDim.x - 5 )
        {
            currentPathOmittingData[idxOfPoint] = true;
            return;
        }
    }
    __syncthreads();

    if(currentPathOmittingData[idxOfPoint])
        return;
    __syncthreads();

    const PathPoint& currentPathPoint = currentPathData[idxOfPoint];
    if(isPointOnIllegalImageBoundary(idxOfPoint, currentPathData, width, height))
    {
        currentPathOmittingData[idxOfPoint] = true;
        return;
    }
    __syncthreads();

    const Polygon::cord_type* pointA = getCoordinatesOfPathPoint(idxOfPoint, currentPathData, coordinateData,
                                                                 width);
    const int idxOfPointB = getIdxOfRelativePoint(idxOfPoint, 1, pathLength);
    const int idxOfPointC = getIdxOfRelativePoint(idxOfPoint, 2, pathLength);
    const int idxOfPointD = getIdxOfRelativePoint(idxOfPoint, 3, pathLength);
    const int idxOfPointE = getIdxOfRelativePoint(idxOfPoint, 4, pathLength);
    const Polygon::cord_type* pointB = getCoordinatesOfPathPoint(idxOfPointB, currentPathData, coordinateData, width);
    const Polygon::cord_type* pointC = getCoordinatesOfPathPoint(idxOfPointC, currentPathData, coordinateData, width);
    const Polygon::cord_type* pointD = getCoordinatesOfPathPoint(idxOfPointD, currentPathData, coordinateData, width);
    const Polygon::cord_type* pointE = getCoordinatesOfPathPoint(idxOfPointE, currentPathData, coordinateData, width);

    bool shouldThisPointBeOmitted = isPointOnIllegalImageBoundary(idxOfPointB, currentPathData, width, height) ||
                                    isPointOnIllegalImageBoundary(idxOfPointC, currentPathData, width, height) ||
                                    isPointOnIllegalImageBoundary(idxOfPointD, currentPathData, width, height) ||
                                    isPointOnIllegalImageBoundary(idxOfPointE, currentPathData, width, height);

    int cornerPatternIdx = 0;
    while(!shouldThisPointBeOmitted && cornerPatternIdx < 25)
    {
        const CornerPatternEdge& step1 = cornerPatterns[4*cornerPatternIdx + 0];
        const CornerPatternEdge& step2 = cornerPatterns[4*cornerPatternIdx + 1];
        const CornerPatternEdge& step3 = cornerPatterns[4*cornerPatternIdx + 2];
        const CornerPatternEdge& step4 = cornerPatterns[4*cornerPatternIdx + 3];
        const CornerPatternEdge step1Reversed = reversePatternEdge(step1);
        const CornerPatternEdge step2Reversed = reversePatternEdge(step2);
        const CornerPatternEdge step3Reversed = reversePatternEdge(step3);
        const CornerPatternEdge step4Reversed = reversePatternEdge(step4);
        shouldThisPointBeOmitted = doesAPatternExist(pointA, pointB, pointC, pointD, pointE,
                                                     step1, step2, step3, step4);

        if(!shouldThisPointBeOmitted)
        {
            shouldThisPointBeOmitted = doesAPatternExist(pointA, pointB, pointC, pointD, pointE,
                                                         step4Reversed, step3Reversed, step2Reversed, step1Reversed);
        }
        if(!shouldThisPointBeOmitted)
        {
            shouldThisPointBeOmitted = doesAPatternExist(pointA, pointB, pointC, pointD, pointE,
                                                         step1Reversed, step2Reversed, step3Reversed, step4Reversed);
        }
        if(!shouldThisPointBeOmitted)
        {
            shouldThisPointBeOmitted = doesAPatternExist(pointE, pointD, pointC, pointB, pointA,
                                                         step1, step2, step3, step4);
        }
        ++cornerPatternIdx;
    }

    if(shouldThisPointBeOmitted)
    {
        currentPathOmittingData[idxOfPoint] = true;
        currentPathOmittingData[idxOfPointB] = true;
        currentPathOmittingData[idxOfPointC] = true;
        currentPathOmittingData[idxOfPointD] = true;
        currentPathOmittingData[idxOfPointE] = true;
    }
}

__device__ bool CurveControlPointExcluding::isPointOnIllegalImageBoundary(int idxOfPoint, const PathPoint* pathData,
                                                                          unsigned int width, unsigned int height)
{
    const PathPoint& currentPathPoint = pathData[idxOfPoint];
    const int& coordinateDataRow = currentPathPoint.rowOfCoordinates;
    const int& coordinateDataCol = currentPathPoint.colOfCoordinates;
    return (coordinateDataRow == height || coordinateDataCol == width || coordinateDataRow == 0 || coordinateDataCol == 0);
}

__device__ bool CurveControlPointExcluding::doesTheEdgeFitThePattern(const Polygon::cord_type* pointA,
                                                                     const Polygon::cord_type* pointB,
                                                                     const CornerPatternEdge& expectedEdge)
{
    int rowDiff = static_cast<int>(pointB[0]) - static_cast<int>(pointA[0]);
    int colDiff = static_cast<int>(pointB[1]) - static_cast<int>(pointA[1]);

    int expectedRowDiff = getRowDiffValueOfPattern(expectedEdge);
    int expectedColDiff = getColDiffValueOfPattern(expectedEdge);

    return ( (rowDiff == expectedRowDiff) && (colDiff == expectedColDiff) );
}

__device__ CornerPatternEdge CurveControlPointExcluding::reversePatternEdge(const CornerPatternEdge& patternEdge)
{
    CornerPatternEdge result = CornerPatternEdge::NOTHING;
    switch(patternEdge)
    {
        case CornerPatternEdge::LONG_VERTICAL_UP:                    result = CornerPatternEdge::LONG_VERTICAL_DOWN; break;
        case CornerPatternEdge::LONG_VERTICAL_DOWN:                  result = CornerPatternEdge::LONG_VERTICAL_UP; break;
        case CornerPatternEdge::LONG_HORIZONTAL_LEFT:                result = CornerPatternEdge::LONG_HORIZONTAL_RIGHT; break;
        case CornerPatternEdge::LONG_HORIZONTAL_RIGHT:               result = CornerPatternEdge::LONG_HORIZONTAL_LEFT; break;
        case CornerPatternEdge::SHORT_VERTICAL_UP:                   result = CornerPatternEdge::SHORT_VERTICAL_DOWN; break;
        case CornerPatternEdge::SHORT_VERTICAL_DOWN:                 result = CornerPatternEdge::SHORT_VERTICAL_UP; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_LEFT:               result = CornerPatternEdge::SHORT_HORIZONTAL_RIGHT; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_RIGHT:              result = CornerPatternEdge::SHORT_HORIZONTAL_LEFT; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_UP:     result = CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_DOWN; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_DOWN:   result = CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_UP; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_UP:      result = CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_DOWN; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_DOWN:    result = CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_UP; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP:   result = CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN: result = CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_UP:    result = CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN:  result = CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_UP; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_UP:             result = CornerPatternEdge::RIGHT_SHORT_DIAGONAL_DOWN; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_DOWN:           result = CornerPatternEdge::RIGHT_SHORT_DIAGONAL_UP; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_UP:              result = CornerPatternEdge::LEFT_SHORT_DIAGONAL_DOWN; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_DOWN:            result = CornerPatternEdge::LEFT_SHORT_DIAGONAL_UP; break;
        default: break;
    }

    return result;
}

__device__ int CurveControlPointExcluding::getRowDiffValueOfPattern(const CornerPatternEdge& patternEdge)
{
    int result = 0;
    switch(patternEdge)
    {
        case CornerPatternEdge::LONG_VERTICAL_UP:                    result = -100; break;
        case CornerPatternEdge::LONG_VERTICAL_DOWN:                  result = 100; break;
        case CornerPatternEdge::LONG_HORIZONTAL_LEFT:                result = 0; break;
        case CornerPatternEdge::LONG_HORIZONTAL_RIGHT:               result = 0; break;
        case CornerPatternEdge::SHORT_VERTICAL_UP:                   result = -50; break;
        case CornerPatternEdge::SHORT_VERTICAL_DOWN:                 result = 50; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_LEFT:               result = 0; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_RIGHT:              result = 0; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_UP:     result = -75; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_DOWN:   result = 75; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_UP:      result = -75; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_DOWN:    result = 75; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP:   result = -25; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN: result = 25; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_UP:    result = -25; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN:  result = 25; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_UP:             result = -50; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_DOWN:           result = 50; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_UP:              result = -50; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_DOWN:            result = 50; break;
        default: break;
    }
        
    return result;
}

__device__ int CurveControlPointExcluding::getColDiffValueOfPattern(const CornerPatternEdge& patternEdge)
{
    int result = 0;
    switch(patternEdge)
    {
        case CornerPatternEdge::LONG_VERTICAL_UP:                    result = 0; break;
        case CornerPatternEdge::LONG_VERTICAL_DOWN:                  result = 0; break;
        case CornerPatternEdge::LONG_HORIZONTAL_LEFT:                result = -100; break;
        case CornerPatternEdge::LONG_HORIZONTAL_RIGHT:               result = 100; break;
        case CornerPatternEdge::SHORT_VERTICAL_UP:                   result = 0; break;
        case CornerPatternEdge::SHORT_VERTICAL_DOWN:                 result = 0; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_LEFT:               result = -50; break;
        case CornerPatternEdge::SHORT_HORIZONTAL_RIGHT:              result = 50; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_UP:     result = 25; break;
        case CornerPatternEdge::VERTICAL_RIGHT_LONG_DIAGONAL_DOWN:   result = -25; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_UP:      result = -25; break;
        case CornerPatternEdge::VERTICAL_LEFT_LONG_DIAGONAL_DOWN:    result = 25; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP:   result = 75; break;
        case CornerPatternEdge::HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN: result = -75; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_UP:    result = -75; break;
        case CornerPatternEdge::HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN:  result = 75; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_UP:             result = 50; break;
        case CornerPatternEdge::RIGHT_SHORT_DIAGONAL_DOWN:           result = -50; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_UP:              result = -50; break;
        case CornerPatternEdge::LEFT_SHORT_DIAGONAL_DOWN:            result = 50; break;
        default: break;
    }

    return result;
}

__device__ bool
CurveControlPointExcluding::doesAPatternExist(const Polygon::cord_type* pointA, const Polygon::cord_type* pointB,
                                              const Polygon::cord_type* pointC, const Polygon::cord_type* pointD,
                                              const Polygon::cord_type* pointE, const CornerPatternEdge& step1,
                                              const CornerPatternEdge& step2, const CornerPatternEdge& step3,
                                              const CornerPatternEdge& step4)
{
    if( !doesTheEdgeFitThePattern(pointA, pointB, step1) ) return false;
    if( !doesTheEdgeFitThePattern(pointB, pointC, step2) ) return false;
    if( !doesTheEdgeFitThePattern(pointC, pointD, step3) ) return false;

    if(step4 == CornerPatternEdge::NOTHING)
        return true;
    else
        return doesTheEdgeFitThePattern(pointD, pointE, step4);
}

__device__ int CurveControlPointExcluding::getIdxOfRelativePoint(const int& source, int steps, const int& pathLength)
{
    int result = source + steps;

    if(steps < 0 && result < 0)
        result += pathLength;
    else if(steps > 0 && result >= pathLength)
        result -= pathLength;

    return result;
}

__device__ const Polygon::cord_type*
CurveControlPointExcluding::getCoordinatesOfPathPoint(const int& pointIdx, const PathPoint* pathData, PolygonSide* coordinateData,
                                                      unsigned int width)
{
    const PathPoint& currentPathPoint = pathData[pointIdx];
    const int& coordinateDataRow = currentPathPoint.rowOfCoordinates;
    const int& coordinateDataCol = currentPathPoint.colOfCoordinates;

    unsigned int coordinateIdx = coordinateDataCol + coordinateDataRow * width;
    if( currentPathPoint.useBPoint )
        return coordinateData[coordinateIdx].pointB;
    else
        return coordinateData[coordinateIdx].pointA;
}

void CurveControlPointExcluding::allocateCornerPatternData()
{
    using CE = CornerPatternEdge;
    CE definedPatterns[25*4] = {
        CE::LONG_VERTICAL_UP, CE::LONG_HORIZONTAL_RIGHT, CE::LONG_VERTICAL_DOWN, CE::NOTHING,
        CE::LONG_HORIZONTAL_LEFT, CE::LONG_VERTICAL_UP, CE::LONG_HORIZONTAL_RIGHT, CE::NOTHING,
        CE::LONG_VERTICAL_DOWN, CE::LONG_HORIZONTAL_LEFT, CE::LONG_VERTICAL_UP, CE::NOTHING,
        CE::LONG_HORIZONTAL_RIGHT, CE::LONG_VERTICAL_UP, CE::LONG_HORIZONTAL_LEFT, CE::NOTHING,
        CE::VERTICAL_LEFT_LONG_DIAGONAL_UP, CE::SHORT_HORIZONTAL_LEFT, CE::VERTICAL_RIGHT_LONG_DIAGONAL_DOWN, CE::NOTHING,
        CE::HORIZONTAL_LEFT_LONG_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP, CE::NOTHING,
        CE::VERTICAL_LEFT_LONG_DIAGONAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::VERTICAL_RIGHT_LONG_DIAGONAL_UP, CE::NOTHING,
        CE::HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN, CE::SHORT_VERTICAL_DOWN, CE::HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN, CE::NOTHING,
        CE::RIGHT_SHORT_DIAGONAL_UP, CE::SHORT_HORIZONTAL_RIGHT, CE::VERTICAL_LEFT_LONG_DIAGONAL_DOWN, CE::NOTHING,
        CE::VERTICAL_RIGHT_LONG_DIAGONAL_UP, CE::SHORT_HORIZONTAL_RIGHT, CE::LEFT_SHORT_DIAGONAL_DOWN, CE::NOTHING,
        CE::LEFT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP, CE::NOTHING,
        CE::HORIZONTAL_LEFT_LONG_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::RIGHT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::VERTICAL_LEFT_LONG_DIAGONAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::RIGHT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::LEFT_SHORT_DIAGONAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::VERTICAL_RIGHT_LONG_DIAGONAL_UP, CE::NOTHING,
        CE::HORIZONTAL_RIGHT_LONG_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::LEFT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::RIGHT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::HORIZONTAL_LEFT_LONG_DIAGONAL_UP, CE::NOTHING,
        CE::RIGHT_SHORT_DIAGONAL_UP, CE::SHORT_HORIZONTAL_RIGHT, CE::LEFT_SHORT_DIAGONAL_DOWN, CE::NOTHING,
        CE::LEFT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::RIGHT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::LEFT_SHORT_DIAGONAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::RIGHT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::RIGHT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::LEFT_SHORT_DIAGONAL_UP, CE::NOTHING,
        CE::LEFT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::SHORT_HORIZONTAL_RIGHT, CE::LEFT_SHORT_DIAGONAL_DOWN,
        CE::RIGHT_SHORT_DIAGONAL_UP, CE::SHORT_VERTICAL_UP, CE::SHORT_HORIZONTAL_LEFT, CE::RIGHT_SHORT_DIAGONAL_DOWN,
        CE::RIGHT_SHORT_DIAGONAL_DOWN, CE::SHORT_VERTICAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::RIGHT_SHORT_DIAGONAL_UP,
        CE::LEFT_SHORT_DIAGONAL_DOWN, CE::SHORT_HORIZONTAL_RIGHT, CE::SHORT_VERTICAL_UP, CE::LEFT_SHORT_DIAGONAL_UP,
        CE::SHORT_HORIZONTAL_RIGHT, CE::SHORT_VERTICAL_UP, CE::SHORT_HORIZONTAL_LEFT, CE::SHORT_VERTICAL_DOWN
    };

    cudaMemcpyToSymbol(cornerPatterns, definedPatterns, 25*4 , 0, cudaMemcpyHostToDevice );
}