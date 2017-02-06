#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CurveControlPointSmoothing.hpp"

__device__ Polygon::cord_type
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

__device__ Polygon::cord_type
CurveControlPointSmoothing::getPositionalEnergy(const ControlPoint& pointA, const ControlPoint& pointB)
{
    Polygon::cord_type rowDistance = 0.01*(pointB.row - pointA.row);
    Polygon::cord_type colDistance = 0.01*(pointB.col - pointA.col);
    Polygon::cord_type combinedDistance = rowDistance*rowDistance + colDistance*colDistance;
    return combinedDistance*combinedDistance;
}

__device__ Polygon::cord_type
CurveControlPointSmoothing::getCurvatureIntegral(const ControlPoint& startPoint, const ControlPoint& midPoint,
                                                 const ControlPoint& endPoint, const int numberOfSamples)
{
    Polygon::cord_type result = 0.0f;
    const Curve::param_type sampleIntervalLength = static_cast<Curve::param_type>(1) / static_cast<Curve::param_type>(numberOfSamples);
    CurveCurvature curvature(startPoint, midPoint, endPoint);

    for(int i = 0; i < numberOfSamples; ++i)
    {
        result += sampleIntervalLength * curvature(sampleIntervalLength * (0.5f + i));
    }

    return result;
}

__device__ int CurveControlPointSmoothing::getIdxOfRelativePoint(const int& source, int steps, const int& pathLength)
{
    int result = source + steps;

    if(steps < 0 && result < 0)
        result += pathLength;
    else if(steps > 0 && result >= pathLength)
        result -= pathLength;

    return result;
}

__global__ void
CurveControlPointSmoothing::optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData,
                                          const unsigned int pathOffset, unsigned int width, unsigned int height,
                                          Polygon::cord_type* randomShiftPointValues, const Polygon::cord_type radius,
                                          bool* omitPoint, int pathLength)
{
    //TODO make sample number externally configurable
    //TODO make number of guesses externally configurable
    const int GUESSES_PER_POINT = 4;
    const int INTEGRAL_SAMPLES = 16;
    const int PATH_ITERATIONS = 4;

    int idxOfPoint = blockIdx.x * blockDim.x + threadIdx.x;
    if(idxOfPoint >= pathLength)
    {
        return;
    }
    __syncthreads();

    int idxOfPointInBlock = threadIdx.x;

    const PathPoint* currentPathData = pathData + pathOffset;

    extern __shared__ int sMem[];
    Polygon::cord_type* originalRows = (Polygon::cord_type*)sMem;
    Polygon::cord_type* originalCols = (Polygon::cord_type*)&sMem[blockDim.x];

    {
        Polygon::cord_type currentRow = getCoordinateOfPathPoint(idxOfPoint, currentPathData, coordinateData, width,
                                                                 height, 0);
        Polygon::cord_type currentCol = getCoordinateOfPathPoint(idxOfPoint, currentPathData, coordinateData, width,
                                                                 height, 1);
        originalRows[idxOfPointInBlock] = currentRow;
        originalCols[idxOfPointInBlock] = currentCol;
    }
    __syncthreads();

    if(omitPoint[idxOfPoint + pathOffset])
    {
        return;
    }
    __syncthreads();

    int idxOf2xPreviousPoint = getIdxOfRelativePoint(idxOfPointInBlock, -2, pathLength);
    int idxOfPreviousPoint = getIdxOfRelativePoint(idxOfPointInBlock, -1, pathLength);
    int idxOfNextPoint = getIdxOfRelativePoint(idxOfPointInBlock, 1, pathLength);
    int idxOf2xNextPoint = getIdxOfRelativePoint(idxOfPointInBlock, 2, pathLength);

    ControlPoint controlPointToModify{originalRows[idxOfPointInBlock], originalCols[idxOfPointInBlock]};
    ControlPoint prev2xControlPoint{originalRows[idxOf2xPreviousPoint], originalCols[idxOf2xPreviousPoint]};
    ControlPoint prevControlPoint{originalRows[idxOfPreviousPoint], originalCols[idxOfPreviousPoint]};
    ControlPoint nextControlPoint{originalRows[idxOfNextPoint], originalCols[idxOfNextPoint]};
    ControlPoint next2xControlPoint{originalRows[idxOf2xNextPoint], originalCols[idxOf2xNextPoint]};
    ControlPoint prevStartPoint{0.5f * (prevControlPoint.row + controlPointToModify.row),
                                0.5f * (prevControlPoint.col + controlPointToModify.col)};
    ControlPoint nextStartPoint{0.5f * (nextControlPoint.row + controlPointToModify.row),
                                0.5f * (nextControlPoint.col + controlPointToModify.col)};
    ControlPoint prev2xStartPoint{0.5f * (prev2xControlPoint.row + prevControlPoint.row),
                                  0.5f * (prev2xControlPoint.col + prevControlPoint.col)};
    ControlPoint next2xStartPoint{0.5f * (next2xControlPoint.row + nextControlPoint.row),
                                  0.5f * (next2xControlPoint.col + nextControlPoint.col)};
    Polygon::cord_type minimumEnergy =
            getCurvatureIntegral(prev2xStartPoint, prevControlPoint, prevStartPoint, INTEGRAL_SAMPLES) +
            getCurvatureIntegral(prevStartPoint, controlPointToModify, nextStartPoint, INTEGRAL_SAMPLES) +
            getCurvatureIntegral(nextStartPoint, nextControlPoint, next2xStartPoint, INTEGRAL_SAMPLES);

    int k = 0;
    while(k++ < PATH_ITERATIONS)
    {
        ControlPoint newVersionOfControlPoint = controlPointToModify;
        for(int i = 0; i < GUESSES_PER_POINT; ++i)
        {
            ControlPoint possibleNewControlPoint = getRandomPointInNeighborhood(newVersionOfControlPoint, radius,
                                                                                randomShiftPointValues + idxOfPoint +
                                                                                i);
            ControlPoint prevNewStartPoint{0.5f * (prevControlPoint.row + possibleNewControlPoint.row),
                                           0.5f * (prevControlPoint.col + possibleNewControlPoint.col)};
            ControlPoint nextNewStartPoint{0.5f * (nextControlPoint.row + possibleNewControlPoint.row),
                                           0.5f * (nextControlPoint.col + possibleNewControlPoint.col)};

            Polygon::cord_type smoothnessEnergyToCompare =
                    getCurvatureIntegral(prev2xStartPoint, prevControlPoint, prevNewStartPoint, INTEGRAL_SAMPLES) +
                    getCurvatureIntegral(prevNewStartPoint, possibleNewControlPoint, nextNewStartPoint, INTEGRAL_SAMPLES) +
                    getCurvatureIntegral(nextNewStartPoint, nextControlPoint, next2xStartPoint, INTEGRAL_SAMPLES);
            Polygon::cord_type positionalEnergy = getPositionalEnergy(controlPointToModify, possibleNewControlPoint);

            if((smoothnessEnergyToCompare + positionalEnergy) < minimumEnergy)
            {
                newVersionOfControlPoint = possibleNewControlPoint;
                minimumEnergy = smoothnessEnergyToCompare + positionalEnergy;
            }
        }
        __syncthreads();

        originalRows[idxOfPointInBlock] = newVersionOfControlPoint.row;
        originalCols[idxOfPointInBlock] = newVersionOfControlPoint.col;
        __syncthreads();

        controlPointToModify = ControlPoint{originalRows[idxOfPointInBlock], originalCols[idxOfPointInBlock]};
        prev2xControlPoint = ControlPoint{originalRows[idxOf2xPreviousPoint], originalCols[idxOf2xPreviousPoint]};
        prevControlPoint = ControlPoint{originalRows[idxOfPreviousPoint], originalCols[idxOfPreviousPoint]};
        nextControlPoint = ControlPoint{originalRows[idxOfNextPoint], originalCols[idxOfNextPoint]};
        next2xControlPoint = ControlPoint{originalRows[idxOf2xNextPoint], originalCols[idxOf2xNextPoint]};
        prev2xStartPoint = ControlPoint{0.5f * (prev2xControlPoint.row + prevControlPoint.row),
                                        0.5f * (prev2xControlPoint.col + prevControlPoint.col)};
        next2xStartPoint = ControlPoint{0.5f * (next2xControlPoint.row + nextControlPoint.row),
                                        0.5f * (next2xControlPoint.col + nextControlPoint.col)};
    }
    __syncthreads();


    setNewCoordinatesOfPathPoint(idxOfPoint, currentPathData, coordinateData, width,
                                 height, originalRows[idxOfPointInBlock], originalCols[idxOfPointInBlock]);
}

__device__ ControlPoint
CurveControlPointSmoothing::getRandomPointInNeighborhood(const ControlPoint& source, const Polygon::cord_type& radius,
                                                         Polygon::cord_type* randomShiftPointValues)
{
    float rowMod = randomShiftPointValues[0];
    float colMod = randomShiftPointValues[1];

    ControlPoint result{source.row + rowMod, source.col + colMod};

    return result;
}

__device__ void
CurveControlPointSmoothing::setNewCoordinatesOfPathPoint(const int& pointIdx, const PathPoint* pathData,
                                                         PolygonSide* coordinateData, unsigned int width,
                                                         unsigned int height, Polygon::cord_type resultRow,
                                                         Polygon::cord_type resultCol)
{
    const PathPoint& currentPathPoint = pathData[pointIdx];
    const int& coordinateDataRow = currentPathPoint.rowOfCoordinates;
    const int& coordinateDataCol = currentPathPoint.colOfCoordinates;
    if(coordinateDataRow != height && coordinateDataCol != width)
    {
        unsigned int coordinateIdx = coordinateDataCol + coordinateDataRow * width;
        if( currentPathPoint.useBPoint )
        {
            coordinateData[coordinateIdx].pointB[0] = resultRow;
            coordinateData[coordinateIdx].pointB[1] = resultCol;
        }
        else
        {
            coordinateData[coordinateIdx].pointA[0] = resultRow;
            coordinateData[coordinateIdx].pointA[1] = resultCol;
        }
    }
}