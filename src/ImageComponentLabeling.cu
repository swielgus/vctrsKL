#include <cstdio>
#include "ImageComponentLabeling.hpp"

__device__ bool
ImageComponentLabeling::areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV,
                                            const Color::byte& bY, const Color::byte& bU, const Color::byte& bV)
{
    const Color::byte thresholdY = 48;
    const Color::byte thresholdU = 7;
    const Color::byte thresholdV = 6;

    return abs(aY - bY) <= thresholdY && abs(aU - bU) <= thresholdU && abs(aV - bV) <= thresholdV;
}

__device__ int ImageComponentLabeling::findRootOfNodeLabel(int* labels, int current)
{
    int nextLabel;
    do
    {
        nextLabel = current;
        current = labels[nextLabel];
    } while (current < nextLabel);
    return current;
}

__device__ void
ImageComponentLabeling::doUnionOfTrees(int* labels, int labelA, int labelB, int* didAnyLabelChange)
{
    int rootOfA = findRootOfNodeLabel(labels, labelA);
    int rootOfB = findRootOfNodeLabel(labels, labelB);

    if( rootOfA > rootOfB )
    {
        atomicMin(labels+rootOfA, rootOfB);
        didAnyLabelChange[0] = 1;
    }
    else if( rootOfB > rootOfA )
    {
        atomicMin(labels+rootOfB, rootOfA);
        didAnyLabelChange[0] = 1;
    }
}

__device__ void
ImageComponentLabeling::checkAndCombineTwoPixelRoots(int* labels, int labelA, int labelB, Color::byte* colorY,
                                                     Color::byte* colorU, Color::byte* colorV, int* didAnyLabelChange,
                                                     int idxLimit)
{
    if(labelA >= 0 && labelA < idxLimit && labelB >= 0 && labelB < idxLimit)
    {
        Color::byte currentY = colorY[labelA];
        Color::byte currentU = colorU[labelA];
        Color::byte currentV = colorV[labelA];
        Color::byte comparedY = colorY[labelB];
        Color::byte comparedU = colorU[labelB];
        Color::byte comparedV = colorV[labelB];
        if(areYUVColorsSimilar(currentY, currentU, currentV, comparedY, comparedU, comparedV))
        {
            doUnionOfTrees(labels, labelA, labelB, didAnyLabelChange);
        }
    }
}

__global__ void
ImageComponentLabeling::mergeSolutionsOnBlockBorders(Color::byte* colorY, Color::byte* colorU, Color::byte* colorV,
                                                     int* labels, int width, int height, int tileSide)
{
    //local tileX and Y are stored directly in blockIdx.x and blockIdx.x
    //all threads for each block are stored in the z-dir of each block (threadIdx.z)
    int rowIdxOfCurrentTileInsideBlock = threadIdx.x + blockIdx.x * blockDim.x;
    int colIdxOfCurrentTileInsideBlock = threadIdx.y + blockIdx.y * blockDim.y;

    int currentColUsedInHorizontalBorderComparing = colIdxOfCurrentTileInsideBlock * tileSide + threadIdx.z;
    int currentRowUsedInVerticalBorderComparing = rowIdxOfCurrentTileInsideBlock * tileSide + threadIdx.z;

    if(currentRowUsedInVerticalBorderComparing < height && currentColUsedInHorizontalBorderComparing < width)
    {

        //the number of times each thread has to be used to process one border of the tile
        int threadIterations = tileSide / blockDim.z;

        //dimensions of the tile on the next level of the merging scheme
        int nextLevelTileSide = tileSide * blockDim.x;

        __shared__ int wasAnyNodeLabelChanged[1];
        int idxLimit = width*height;

        while(true)
        {
            if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
                wasAnyNodeLabelChanged[0] = 0;

            __syncthreads();

            int distanceOfCurrentTilePixelFromBeginningOfTileLine;
            if(threadIdx.x < blockDim.x - 1) //horizontal borders between tiles only
            {
                int lastRowOfCurrentTile = (rowIdxOfCurrentTileInsideBlock + 1) * tileSide - 1;
                distanceOfCurrentTilePixelFromBeginningOfTileLine = threadIdx.y * tileSide + threadIdx.z;

                if(lastRowOfCurrentTile < height && currentColUsedInHorizontalBorderComparing < width)
                {
                    for(int i = 0; i < threadIterations; ++i)
                    {
                        int idxOfCurrentTilePixel = currentColUsedInHorizontalBorderComparing +
                                                    lastRowOfCurrentTile * width;

                        int idxOfComparedPixel = idxOfCurrentTilePixel + width; //below
                        checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY, colorU,
                                                     colorV, wasAnyNodeLabelChanged, idxLimit);

                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine > 0)
                        //not the element of leftmost tile column
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width - 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY,
                                                         colorU, colorV, wasAnyNodeLabelChanged, idxLimit);
                        }
                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine < nextLevelTileSide - 1)
                        //not the element of rightmost tile column
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY,
                                                         colorU, colorV, wasAnyNodeLabelChanged, idxLimit);
                        }
                        lastRowOfCurrentTile += blockDim.z;
                        distanceOfCurrentTilePixelFromBeginningOfTileLine += blockDim.z;
                    }
                }
            }

            if(threadIdx.y < blockDim.y - 1) //vertical borders between tiles only
            {
                int lastColOfCurrentTile = (colIdxOfCurrentTileInsideBlock + 1) * tileSide - 1;
                distanceOfCurrentTilePixelFromBeginningOfTileLine = threadIdx.x * tileSide + threadIdx.z;

                if(lastColOfCurrentTile < width && currentRowUsedInVerticalBorderComparing < height)
                {
                    for(int i = 0; i < threadIterations; ++i)
                    {
                        int idxOfCurrentTilePixel = lastColOfCurrentTile +
                                                    currentRowUsedInVerticalBorderComparing * width;

                        int idxOfComparedPixel = idxOfCurrentTilePixel + 1; //right
                        checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY, colorU,
                                                     colorV, wasAnyNodeLabelChanged, idxLimit);

                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine > 0)
                        //not the element of uppermost tile row
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel - width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY,
                                                         colorU, colorV, wasAnyNodeLabelChanged, idxLimit);
                        }
                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine < nextLevelTileSide - 1)
                        //not the element of lowermost tile row
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorY,
                                                         colorU, colorV, wasAnyNodeLabelChanged, idxLimit);
                        }
                        lastColOfCurrentTile += blockDim.z;
                        distanceOfCurrentTilePixelFromBeginningOfTileLine += blockDim.z;
                    }
                }
            }
            __syncthreads();

            if(wasAnyNodeLabelChanged[0] == 0)
                break;

            //need to synchronize here because the wasAnyNodeLabelChanged variable is changed next
            __syncthreads();
        }
    }
}

__global__ void
ImageComponentLabeling::createLocalComponentLabels(Color::byte* colorY, Color::byte* colorU, Color::byte* colorV,
                                                   int* output, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < height && col < width)
    {
        int blockSide = blockDim.x;
        int idxInSharedBlock = threadIdx.y + (threadIdx.x) * blockSide;
        int idx = col + row * width;
        int newLabel = idxInSharedBlock;
        int oldLabel = 0;

        extern __shared__ int sMem[];
        int numberOfPixelsPerBlock = blockDim.x * blockDim.y;
        int* sharedLabels = sMem;
        Color::byte* sharedYData = (Color::byte*)&sMem[numberOfPixelsPerBlock];
        Color::byte* sharedUData = &sharedYData[numberOfPixelsPerBlock];
        Color::byte* sharedVData = &sharedUData[numberOfPixelsPerBlock];

        __shared__ int didAnyLabelChange[1];

        Color::byte currentY = colorY[idx];
        Color::byte currentU = colorU[idx];
        Color::byte currentV = colorV[idx];

        sharedYData[idxInSharedBlock] = currentY;
        sharedUData[idxInSharedBlock] = currentU;
        sharedVData[idxInSharedBlock] = currentV;
        __syncthreads();

        while(true)
        {
            sharedLabels[idxInSharedBlock] = newLabel;

            if((threadIdx.x | threadIdx.y) == 0)
                didAnyLabelChange[0] = 0;

            oldLabel = newLabel;
            __syncthreads();

            for(int modRow = -1; modRow <= 1; ++modRow)
            for(int modCol = -1; modCol <= 1; ++modCol)
            {
                int currentRowInSharedBlock = threadIdx.x + modRow;
                int currentColInSharedBlock = threadIdx.y + modCol;
                int currentRowInWholeImage = row + modRow;
                int currentColInWholeImage = col + modCol;
                if( (modRow != 0 || modCol != 0) &&
                    (currentRowInSharedBlock >= 0 && currentRowInSharedBlock < blockSide) &&
                    (currentColInSharedBlock >= 0 && currentColInSharedBlock < blockSide) &&
                    (currentRowInWholeImage >= 0 && currentRowInWholeImage < height) &&
                    (currentColInWholeImage >= 0 && currentColInWholeImage < width) )
                {
                    int currentIdxInSharedBlock = currentColInSharedBlock + currentRowInSharedBlock * blockSide;
                    if( areYUVColorsSimilar(currentY, currentU, currentV,
                                            sharedYData[currentIdxInSharedBlock],
                                            sharedUData[currentIdxInSharedBlock],
                                            sharedVData[currentIdxInSharedBlock]) )
                        newLabel = min(newLabel, sharedLabels[currentIdxInSharedBlock]);
                }
            }
            __syncthreads();

            if(oldLabel > newLabel)
            {
                atomicMin(sharedLabels + oldLabel, newLabel);
                didAnyLabelChange[0] = 1;
            }
            __syncthreads();

            if(didAnyLabelChange[0] == 0)
                break;

            newLabel = findRootOfNodeLabel(sharedLabels,newLabel);
            __syncthreads();
        }

        row = newLabel / (blockSide);
        col = newLabel - row * (blockSide);
        col = blockIdx.y * blockDim.y + col;
        row = blockIdx.x * blockDim.x + row;
        newLabel = col + row * width;
        output[idx] = newLabel;
    }
}

__global__ void ImageComponentLabeling::flattenAllEquivalenceTrees(int* labels, int width, int height)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row < height && col < width)
    {
        int idx = col + row * width;
        int labelOfCurrentElement = labels[idx];
        if(labelOfCurrentElement != idx)
        {
            int newLabel = findRootOfNodeLabel(labels, labelOfCurrentElement);
            if(newLabel < labelOfCurrentElement)
            {
                labels[idx] = newLabel;
            }
        }
    }
}