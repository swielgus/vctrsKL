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
ImageComponentLabeling::checkAndCombineTwoPixelRoots(int* labels, int labelA, int labelB, Color::byte* colorYUV,
                                                     int* didAnyLabelChange, int idxLimit)
{
    if(labelA >= 0 && labelA < idxLimit && labelB >= 0 && labelB < idxLimit)
    {
        Color::byte currentY = colorYUV[3*labelA+0];
        Color::byte currentU = colorYUV[3*labelA+1];
        Color::byte currentV = colorYUV[3*labelA+2];
        Color::byte comparedY = colorYUV[3*labelB+0];
        Color::byte comparedU = colorYUV[3*labelB+1];
        Color::byte comparedV = colorYUV[3*labelB+2];
        if(areYUVColorsSimilar(currentY, currentU, currentV, comparedY, comparedU, comparedV))
        {
            doUnionOfTrees(labels, labelA, labelB, didAnyLabelChange);
        }
    }
}

__global__ void
ImageComponentLabeling::mergeSolutionsOnBlockBorders(Color::byte* colorYUV, int* labels, int width, int height, int tileSide)
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
                        checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                     wasAnyNodeLabelChanged, idxLimit);

                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine > 0)
                        //not the element of leftmost tile column
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width - 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                         wasAnyNodeLabelChanged, idxLimit);
                        }
                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine < nextLevelTileSide - 1)
                        //not the element of rightmost tile column
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                         wasAnyNodeLabelChanged, idxLimit);
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
                        checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                     wasAnyNodeLabelChanged, idxLimit);

                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine > 0)
                        //not the element of uppermost tile row
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel - width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                         wasAnyNodeLabelChanged, idxLimit);
                        }
                        if(distanceOfCurrentTilePixelFromBeginningOfTileLine < nextLevelTileSide - 1)
                        //not the element of lowermost tile row
                        {
                            idxOfComparedPixel = idxOfCurrentTilePixel + width + 1;
                            checkAndCombineTwoPixelRoots(labels, idxOfCurrentTilePixel, idxOfComparedPixel, colorYUV,
                                                         wasAnyNodeLabelChanged, idxLimit);
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
ImageComponentLabeling::createLocalComponentLabels(Color::byte* colorYUV, int* output, int width, int height)
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
        Color::byte* sharedYUVData = (Color::byte*)&sMem[numberOfPixelsPerBlock];

        __shared__ int didAnyLabelChange[1];

        Color::byte currentY = colorYUV[3*idx+0];
        Color::byte currentU = colorYUV[3*idx+1];
        Color::byte currentV = colorYUV[3*idx+2];

        sharedYUVData[3*idxInSharedBlock+0] = currentY;
        sharedYUVData[3*idxInSharedBlock+1] = currentU;
        sharedYUVData[3*idxInSharedBlock+2] = currentV;
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
                                            sharedYUVData[3*currentIdxInSharedBlock+0],
                                            sharedYUVData[3*currentIdxInSharedBlock+1],
                                            sharedYUVData[3*currentIdxInSharedBlock+2]) )
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

void ImageComponentLabeling::setComponentLabels(Color::byte* d_colorYUVData, int* d_componentLabels, unsigned int width, unsigned int height)
{
    int blockSide = 16;
    dim3 dimBlock(blockSide, blockSide);
    dim3 dimGrid((height + dimBlock.y - 1)/dimBlock.y, (width + dimBlock.x - 1)/dimBlock.x);

    int numberOfPixelsPerBlock = dimBlock.x * dimBlock.y;

    //solve components in local squares
    ImageComponentLabeling::createLocalComponentLabels <<<dimGrid, dimBlock, (numberOfPixelsPerBlock * sizeof(int))+(3 * numberOfPixelsPerBlock * sizeof(Color::byte))>>>(
            d_colorYUVData, d_componentLabels, width, height);
    cudaDeviceSynchronize();

    //merge small results into bigger groups
    while( (blockSide < width || blockSide < height) )
    {
        //compute the number of tiles that are going to be merged in a single thread block
        int numberOfTileRows = 4;
        int numberOfTileCols = 4;

        if(numberOfTileCols * blockSide > width)
            numberOfTileCols = (width + blockSide - 1) / blockSide;

        if(numberOfTileRows * blockSide > height)
            numberOfTileRows = (height + blockSide - 1) / blockSide;

        int threadsPerTile = 32;
        if(blockSide < threadsPerTile)
            threadsPerTile = blockSide;

        dim3 block(numberOfTileRows, numberOfTileCols, threadsPerTile);
        dim3 grid((height + (numberOfTileRows * blockSide) - 1) / (numberOfTileRows * blockSide),
                  (width + (numberOfTileCols * blockSide) - 1) / (numberOfTileCols * blockSide));
        ImageComponentLabeling::mergeSolutionsOnBlockBorders<<<grid, block>>>(
                d_colorYUVData, d_componentLabels, width, height, blockSide);

        if(numberOfTileCols > numberOfTileRows)
            blockSide = numberOfTileCols * blockSide;
        else
            blockSide = numberOfTileRows * blockSide;

        //TODO (opt) update labels on borders
    }

    //update all labels
    ImageComponentLabeling::flattenAllEquivalenceTrees<<<dimGrid, dimBlock>>>(d_componentLabels, width, height);
}