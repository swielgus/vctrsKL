#include <algorithm>
#include <iostream>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveOptimizer.hpp"
#include "CurveControlPointSmoothing.hpp"

CurveOptimizer::CurveOptimizer(PolygonSideMap& usedSideMap)
    : imageWidth{usedSideMap.getImageWidth()}, imageHeight{usedSideMap.getImageHeight()}, d_coordinateData{nullptr},
      usedPathPoints{}, pathAddressOffsets{}, d_pathPointData{nullptr}
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
    std::random_device rd;
    std::mt19937 gen(rd());

    //TODO make the neighborhood radius externally configurable
    const Polygon::cord_type RADIUS = 12;
    const int GUESSES_PER_POINT = 4;
    std::uniform_real_distribution<> dis(-RADIUS, RADIUS);

    for(int pathIdx = 0; pathIdx < usedPathPoints.size(); ++pathIdx)
    {
        const unsigned int numberOfPathPoints = usedPathPoints[pathIdx].size();
        const unsigned int& addressOffsetOfPath = pathAddressOffsets[pathIdx];
        std::vector<Polygon::cord_type> randomShiftPointValues(numberOfPathPoints * 2 * GUESSES_PER_POINT);

        std::generate(randomShiftPointValues.begin(), randomShiftPointValues.end(), [&](){ return dis(gen); });

        //for(int i = 0; i < 4; ++i)
        {
            Polygon::cord_type* d_randomShiftPointValues;
            cudaMalloc( &d_randomShiftPointValues, randomShiftPointValues.size() * sizeof(Polygon::cord_type));
            cudaMemcpy( d_randomShiftPointValues, randomShiftPointValues.data(),
                        randomShiftPointValues.size() * sizeof(Polygon::cord_type),
                        cudaMemcpyHostToDevice );

            CurveControlPointSmoothing::optimizeCurve<<<1, numberOfPathPoints,
                    2 * numberOfPathPoints * sizeof(PolygonSide::point_type)>>>(
                            d_coordinateData, d_pathPointData, addressOffsetOfPath, imageWidth, imageHeight,
                            d_randomShiftPointValues, RADIUS);
            cudaDeviceSynchronize();

            cudaFree(d_randomShiftPointValues);
        }
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