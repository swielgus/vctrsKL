#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveControlPointSmoothing.hpp"
#include "CurveCurvature.hpp"

__device__ PolygonSide::point_type
CurveControlPointSmoothing::getCoordinateOfPathPoint(const int& pointIdx, const PathPoint* pathData, PolygonSide* coordinateData,
                                                     unsigned int width, unsigned int height, int coordinateType)
{
    const PathPoint& currentPathPoint = pathData[pointIdx];
    const int& coordinateDataRow = currentPathPoint.rowOfCoordinates;
    const int& coordinateDataCol = currentPathPoint.colOfCoordinates;
    if(coordinateDataRow == height || coordinateDataCol == width)
    {
        if(coordinateType == 0)
            return static_cast<unsigned int>(coordinateDataRow * 100);
        else
            return static_cast<unsigned int>(coordinateDataCol * 100);
    }

    unsigned int coordinateIdx = coordinateDataCol + coordinateDataRow * width;
    if( currentPathPoint.useBPoint )
        return coordinateData[coordinateIdx].pointB[coordinateType];
    else
        return coordinateData[coordinateIdx].pointA[coordinateType];
}

__global__ void
CurveControlPointSmoothing::optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData,
                                          const unsigned int pathOffset, unsigned int width, unsigned int height)
{
    int pointIdx = threadIdx.x;
    int pathLength = blockDim.x;
    const PathPoint* currentPathData = pathData + pathOffset;

    extern __shared__ int sMem[];
    PolygonSide::point_type* originalRows = (PolygonSide::point_type*)sMem;
    PolygonSide::point_type* originalCols = (PolygonSide::point_type*)&sMem[pathLength];

    {
        PolygonSide::point_type currentRow = getCoordinateOfPathPoint(pointIdx, currentPathData, coordinateData, width,
                                                                      height, 0);
        PolygonSide::point_type currentCol = getCoordinateOfPathPoint(pointIdx, currentPathData, coordinateData, width,
                                                                      height, 1);
        originalRows[pointIdx] = currentRow;
        originalCols[pointIdx] = currentCol;
    }
    __syncthreads();

    int idxOfPreviousPoint = pointIdx - 1;
    if( pointIdx == 0 )
        idxOfPreviousPoint = pathLength - 1;
    int idxOfNextPoint = pointIdx + 1;
    if( pointIdx == pathLength - 1 )
        idxOfNextPoint = 0;

    CurveCurvature curvatureAtPoint(originalRows[idxOfPreviousPoint], originalCols[idxOfPreviousPoint],
                                    originalRows[pointIdx], originalCols[pointIdx],
                                    originalRows[idxOfNextPoint], originalCols[idxOfNextPoint]);

}