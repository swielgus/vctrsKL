#include <algorithm>
#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveOptimizer.hpp"
#include "CurveControlPointSmoothing.hpp"

CurveOptimizer::CurveOptimizer(PolygonSideMap& usedSideMap)
    : imageWidth{usedSideMap.getImageWidth()}, d_coordinateData{nullptr}, usedPathPoints{}, pathAddressOffsets{}, d_pathPointData{nullptr}
{
    d_coordinateData = usedSideMap.getGPUAddressOfPolygonCoordinateData();
    usedPathPoints = std::move(usedSideMap.getPathPointBoundaries());
    allocatePathPointDataOnDevice();
    optimizeEnergyInAllPaths();
}

CurveOptimizer::~CurveOptimizer()
{
    cudaFree(d_pathPointData);
}

void CurveOptimizer::optimizeEnergyInAllPaths()
{
    for(int pathIdx = 0; pathIdx < usedPathPoints.size(); ++pathIdx)
    {
        const unsigned int numberOfPathPoints = usedPathPoints[pathIdx].size();
        const unsigned int& addressOffsetOfPath = pathAddressOffsets[pathIdx];

        CurveControlPointSmoothing::optimizeCurve<<<1, numberOfPathPoints,
                                                    2 * numberOfPathPoints * sizeof(PolygonSide::point_type)>>>(
            d_coordinateData, d_pathPointData, addressOffsetOfPath, imageWidth);
        cudaDeviceSynchronize();
    }
}

void CurveOptimizer::allocatePathPointDataOnDevice()
{
    unsigned int numberOfPathPointsTotal = 0;
    for(const auto& path : usedPathPoints)
    {
        pathAddressOffsets.push_back(numberOfPathPointsTotal);
        numberOfPathPointsTotal += path.size();
    }

    cudaMalloc( &d_pathPointData, numberOfPathPointsTotal * sizeof(PathPoint));

    for(int idx = 0; idx < usedPathPoints.size(); ++idx)
    {
        const auto& path = usedPathPoints[idx];
        const unsigned int numberOfPathPoints = path.size();
        cudaMemcpy( d_pathPointData + pathAddressOffsets[idx], path.data(), numberOfPathPoints * sizeof(PathPoint),
                    cudaMemcpyHostToDevice );
    }
}