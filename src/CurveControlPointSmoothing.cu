#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveControlPointSmoothing.hpp"

__device__ PolygonSide::point_type
CurveControlPointSmoothing::getCoordinateOfPathPoint(const int& pointIdx, const PathPoint* pathData, PolygonSide* coordinateData,
                                                     unsigned int width, int coordinateType)
{
    const PathPoint& currentPathPoint = pathData[pointIdx];
    unsigned int coordinateIdx = currentPathPoint.colOfCoordinates + currentPathPoint.rowOfCoordinates * width;
    if( currentPathPoint.useBPoint )
        return coordinateData[coordinateIdx].pointB[coordinateType];
    else
        return coordinateData[coordinateIdx].pointA[coordinateType];
}

__global__ void
CurveControlPointSmoothing::optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData,
                                          const unsigned int pathOffset, unsigned int width)
{
    int pointIdx = threadIdx.x;
    int pathLength = blockDim.x;
    const PathPoint* currentPathData = pathData + pathOffset;

    extern __shared__ int sMem[];
    PolygonSide::point_type* originalRows = (PolygonSide::point_type*)sMem;
    PolygonSide::point_type* originalCols = (PolygonSide::point_type*)&sMem[pathLength];

    {
        PolygonSide::point_type currentRow = getCoordinateOfPathPoint(pointIdx, currentPathData, coordinateData, width, 0);
        PolygonSide::point_type currentCol = getCoordinateOfPathPoint(pointIdx, currentPathData, coordinateData, width, 1);
        originalRows[pointIdx] = currentRow;
        originalCols[pointIdx] = currentCol;
    }
    __syncthreads();
}