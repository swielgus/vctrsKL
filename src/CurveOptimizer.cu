#include <algorithm>
#include <random>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "CurveOptimizer.hpp"
#include "CurveControlPointSmoothing.hpp"
#include "CurveControlPointExcluding.hpp"

CurveOptimizer::CurveOptimizer(PolygonSideMap& usedSideMap)
    : imageWidth{usedSideMap.getImageWidth()}, imageHeight{usedSideMap.getImageHeight()}, d_coordinateData{nullptr},
      usedPathPoints{}, pathAddressOffsets{}, d_pathPointData{nullptr}, d_omitPointDuringOptimization{nullptr}
{
    d_coordinateData = usedSideMap.getGPUAddressOfPolygonCoordinateData();
    usedPathPoints = std::move(usedSideMap.getPathPointBoundaries());
    allocatePathPointDataOnDevice();
    checkWhichPointsAreToBeOmitted();
    optimizeEnergyInAllPaths();
}

CurveOptimizer::~CurveOptimizer()
{
    cudaFree(d_pathPointData);
    cudaFree(d_omitPointDuringOptimization);
}

void CurveOptimizer::checkWhichPointsAreToBeOmitted()
{
    CurveControlPointExcluding::allocateCornerPatternData();

    for(int pathIdx = 0; pathIdx < usedPathPoints.size(); ++pathIdx)
    {
        const unsigned int numberOfPathPoints = usedPathPoints[pathIdx].size();
        const unsigned int& addressOffsetOfPath = pathAddressOffsets[pathIdx];

        int numberOfThreadsPerBlock = std::min(numberOfPathPoints, 1024U);
        int numberOfBlocksForThisPath = (numberOfPathPoints + numberOfThreadsPerBlock - 1)/numberOfThreadsPerBlock;

        CurveControlPointExcluding::verifyWhichPointsAreToBeIgnored
                <<<numberOfBlocksForThisPath, numberOfThreadsPerBlock>>>
                (d_coordinateData, d_pathPointData, d_omitPointDuringOptimization, addressOffsetOfPath, imageWidth,
                 imageHeight, numberOfPathPoints);
        cudaDeviceSynchronize();
    }
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
        Polygon::cord_type* d_randomShiftPointValues;
        cudaMalloc( &d_randomShiftPointValues, randomShiftPointValues.size() * sizeof(Polygon::cord_type));
        cudaMemcpy( d_randomShiftPointValues, randomShiftPointValues.data(),
                    randomShiftPointValues.size() * sizeof(Polygon::cord_type),
                    cudaMemcpyHostToDevice );

        int numberOfThreadsPerBlock = std::min(numberOfPathPoints, 1024U);
        int numberOfBlocksForThisPath = (numberOfPathPoints + numberOfThreadsPerBlock - 1)/numberOfThreadsPerBlock;

        CurveControlPointSmoothing::optimizeCurve<<<numberOfBlocksForThisPath, numberOfThreadsPerBlock,
                2 * numberOfThreadsPerBlock * sizeof(PolygonSide::point_type)>>>(
                        d_coordinateData, d_pathPointData, addressOffsetOfPath, imageWidth, imageHeight,
                        d_randomShiftPointValues, RADIUS, d_omitPointDuringOptimization, numberOfPathPoints);
        cudaDeviceSynchronize();

        cudaFree(d_randomShiftPointValues);
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

    cudaMalloc( &d_omitPointDuringOptimization, numberOfPathPointsTotal * sizeof(bool));
    bool* tempOmitArray = new bool[numberOfPathPointsTotal];
    for(int i = 0; i < numberOfPathPointsTotal; ++i) tempOmitArray[i] = false;
    cudaMemcpy( d_omitPointDuringOptimization, tempOmitArray, numberOfPathPointsTotal * sizeof(bool),
                cudaMemcpyHostToDevice );
    delete[] tempOmitArray;

    for(int idx = 0; idx < usedPathPoints.size(); ++idx)
    {
        const auto& path = usedPathPoints[idx];
        const unsigned int numberOfPathPoints = path.size();
        cudaMemcpy( d_pathPointData + pathAddressOffsets[idx], path.data(), numberOfPathPoints * sizeof(PathPoint),
                    cudaMemcpyHostToDevice );
    }
}