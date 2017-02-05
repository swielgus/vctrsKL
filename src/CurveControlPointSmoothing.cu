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

__global__ void
CurveControlPointSmoothing::optimizeCurve(PolygonSide* coordinateData, const PathPoint* pathData,
                                          const unsigned int pathOffset, unsigned int width, unsigned int height,
                                          Polygon::cord_type* randomShiftPointValues, const Polygon::cord_type radius)
{
    //TODO make sample number externally configurable
    //TODO make number of guesses externally configurable
    const int GUESSES_PER_POINT = 4;
    const int INTEGRAL_SAMPLES = 16;
    const int PATH_ITERATIONS = 4;

    int idxOfPoint = threadIdx.x;
    int pathLength = blockDim.x;
    const PathPoint* currentPathData = pathData + pathOffset;

    extern __shared__ int sMem[];
    Polygon::cord_type* originalRows = (Polygon::cord_type*)sMem;
    Polygon::cord_type* originalCols = (Polygon::cord_type*)&sMem[pathLength];

    {
        Polygon::cord_type currentRow = getCoordinateOfPathPoint(idxOfPoint, currentPathData, coordinateData, width,
                                                                 height, 0);
        Polygon::cord_type currentCol = getCoordinateOfPathPoint(idxOfPoint, currentPathData, coordinateData, width,
                                                                 height, 1);
        originalRows[idxOfPoint] = currentRow;
        originalCols[idxOfPoint] = currentCol;
    }

    int idxOf2xPreviousPoint = idxOfPoint - 2;
    if(idxOf2xPreviousPoint < 0) idxOf2xPreviousPoint += pathLength;

    int idxOfPreviousPoint = idxOfPoint - 1;
    if(idxOfPreviousPoint < 0) idxOfPreviousPoint += pathLength;

    int idxOfNextPoint = idxOfPoint + 1;
    if(idxOfNextPoint >= pathLength) idxOfNextPoint -= pathLength;

    int idxOf2xNextPoint = idxOfPoint + 2;
    if(idxOf2xNextPoint >= pathLength) idxOf2xNextPoint -= pathLength;
    __syncthreads();

    ControlPoint controlPointToModify{originalRows[idxOfPoint], originalCols[idxOfPoint]};
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

        originalRows[idxOfPoint] = newVersionOfControlPoint.row;
        originalCols[idxOfPoint] = newVersionOfControlPoint.col;
        __syncthreads();

        controlPointToModify = ControlPoint{originalRows[idxOfPoint], originalCols[idxOfPoint]};
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
                                 height, originalRows, originalCols);
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
                                                         unsigned int height, Polygon::cord_type* sharedRows,
                                                         Polygon::cord_type* sharedCols)
{
    const PathPoint& currentPathPoint = pathData[pointIdx];
    const int& coordinateDataRow = currentPathPoint.rowOfCoordinates;
    const int& coordinateDataCol = currentPathPoint.colOfCoordinates;
    if(coordinateDataRow != height && coordinateDataCol != width)
    {
        unsigned int coordinateIdx = coordinateDataCol + coordinateDataRow * width;
        if( currentPathPoint.useBPoint )
        {
            coordinateData[coordinateIdx].pointB[0] = sharedRows[pointIdx];
            coordinateData[coordinateIdx].pointB[1] = sharedCols[pointIdx];
        }
        else
        {
            coordinateData[coordinateIdx].pointA[0] = sharedRows[pointIdx];
            coordinateData[coordinateIdx].pointA[1] = sharedCols[pointIdx];
        }
    }
}