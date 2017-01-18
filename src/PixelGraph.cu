#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "PixelGraph.hpp"
#include "CrossingResolving.h"

__device__ PixelGraph::color_type getColorDifference(const PixelGraph::color_type& a, const PixelGraph::color_type& b)
{
    return (a < b) ? (b-a) : (a-b);
}

__device__ bool areYUVColorsSimilar(const PixelGraph::color_type& aY, const PixelGraph::color_type& aU,
                                  const PixelGraph::color_type& aV, const PixelGraph::color_type& bY,
                                  const PixelGraph::color_type& bU, const PixelGraph::color_type& bV)
{
    const PixelGraph::color_type thresholdY = 48;
    const PixelGraph::color_type thresholdU = 7;
    const PixelGraph::color_type thresholdV = 6;

    //return (getColorDifference(aY, bY) <= thresholdY) && (getColorDifference(aU, bU) <= thresholdU) &&
    //       (getColorDifference(aV, bV) <= thresholdV);
    return abs(aY - bY) <= thresholdY && abs(aU - bU) <= thresholdU && abs(aV - bV) <= thresholdV;
}

__global__ void
createConnections(PixelGraph::color_type* edges, const PixelGraph::color_type* colorY,
                  const PixelGraph::color_type* colorU, const PixelGraph::color_type* colorV,
                  const std::size_t* dim, const PixelGraph::color_type* directions)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    if(i < dim[0] && j < dim[1])
    {
        std::size_t idx = j + i * dim[1];
        edges[idx] = 0;

        for(int iMod = -1; iMod <= 1; ++iMod)
        for(int jMod = -1; jMod <= 1; ++jMod)
        {
            int iNew = i+iMod;
            int jNew = j+jMod;

            if( (iNew != i || jNew != j) && (iNew >= 0 && iNew < dim[0]) && (jNew >= 0 && jNew < dim[1]) )
            {
                std::size_t comparedIdx = iNew * dim[1] + jNew;
                if(areYUVColorsSimilar(colorY[idx],colorU[idx],colorV[idx],
                                       colorY[comparedIdx],colorU[comparedIdx],colorV[comparedIdx]))
                {
                    /* graph directions relative to point x:
                     *  1 | 128 | 64
                     * --------------
                     *  2 |  x  | 32
                     * --------------
                     *  4 |  8  | 16
                     */
                    edges[idx] += directions[(iMod+1)*3 + (jMod+1)];
                }
            }
        }
    }
}

__global__ void resolveCrossings(PixelGraph::color_type* edges, const std::size_t* dim,
                                 const PixelGraph::color_type* directions)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    if(i < dim[0]-1 && j < dim[1]-1)
    {
        std::size_t idx = j + i * dim[1];

        PixelGraph::color_type upperLeftConnected[12] = {255,48,24,255,56,255,255,255,255,40,40,255};
        PixelGraph::color_type lowerLeftConnected[12] = {96,255,255,192,255,255,224,255,255,160,255,160};
        PixelGraph::color_type upperRightConnected[12] = {6,255,255,12,255,14,255,255,10,255,10,255};
        PixelGraph::color_type lowerRightConnected[12] = {255,3,129,255,255,255,255,131,130,255,255,130};

        int k = 0;
        bool squareIsNotConnected = true;
        while(k < 12 && squareIsNotConnected)
        {
            squareIsNotConnected = !(
                    ((upperLeftConnected[k] == 255) || ((upperLeftConnected[k] & edges[idx]) == upperLeftConnected[k]))
                    && ((upperRightConnected[k] == 255) ||
                        ((upperRightConnected[k] & edges[idx + 1]) == upperRightConnected[k]))
                    && ((lowerLeftConnected[k] == 255) ||
                        ((lowerLeftConnected[k] & edges[idx + dim[1]]) == lowerLeftConnected[k]))
                    && ((lowerRightConnected[k] == 255) ||
                        ((lowerRightConnected[k] & edges[idx + dim[1] + 1]) == lowerRightConnected[k]))
            );
            ++k;
        }

        if(!squareIsNotConnected)
        {
            CrossingResolving::doAtomicOr(&edges[idx], directions[5] + directions[7]);
            CrossingResolving::doAtomicOr(&edges[idx+1], directions[3] + directions[7]);
            CrossingResolving::doAtomicOr(&edges[idx+dim[1]], directions[1] + directions[5]);
            CrossingResolving::doAtomicOr(&edges[idx+dim[1]+1], directions[1] + directions[3]);

            CrossingResolving::doAtomicAnd(&edges[idx], ~directions[8]);
            CrossingResolving::doAtomicAnd(&edges[idx+dim[1]+1], ~directions[0]);
            CrossingResolving::doAtomicAnd(&edges[idx+1], ~directions[6]);
            CrossingResolving::doAtomicAnd(&edges[idx+dim[1]], ~directions[2]);
        }



    }
}

PixelGraph::PixelGraph(const ImageData& image)
    : sourceImage{image}, d_pixelConnections{nullptr}, d_pixelDirections{nullptr}
{
    constructGraph();
}

PixelGraph::~PixelGraph()
{
    freeDeviceData();
    //cudaDeviceReset();
}

void PixelGraph::freeDeviceData()
{
    cudaFree(d_pixelConnections);
    cudaFree(d_pixelDirections);
}

void PixelGraph::constructGraph()
{
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    freeDeviceData();

    const PixelGraph::color_type directions[9] = {1,128,64,2,0,32,4,8,16};
    cudaMalloc( &d_pixelDirections, 9 * sizeof(color_type));
    cudaMemcpy( d_pixelDirections, &directions, 9 * sizeof(color_type), cudaMemcpyHostToDevice );

    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();
    cudaMalloc( &d_pixelConnections, width * height * sizeof(color_type));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    //cudaEventRecord(start);
    createConnections<<<dimGrid, dimBlock>>>(d_pixelConnections, sourceImage.getGPUAddressOfYColorData(),
                                    sourceImage.getGPUAddressOfUColorData(), sourceImage.getGPUAddressOfVColorData(),
                                    sourceImage.getGPUAddressOfDimensionsData(), d_pixelDirections);
    cudaDeviceSynchronize();
    //cudaEventRecord(stop);

    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("time:%f\n", milliseconds);
}

std::vector< std::vector<PixelGraph::color_type> > PixelGraph::getEdgeValues() const
{
    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();

    std::vector<std::vector<PixelGraph::color_type>> result;
    result.resize(height);
    for(std::vector<PixelGraph::color_type>& row : result)
        row.resize(width);

    color_type* pixelDirection = new color_type[width * height];
    cudaMemcpy(pixelDirection, d_pixelConnections, width * height * sizeof(color_type), cudaMemcpyDeviceToHost);

    for(std::size_t x = 0; x < height; ++x)
    for(std::size_t y = 0; y < width; ++y)
    {
        result[x][y] = *(pixelDirection + (y + x * width));
    }

    delete[] pixelDirection;

    return result;
}

void PixelGraph::resolveUnnecessaryDiagonals()
{
    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();
    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    resolveCrossings<<<dimGrid, dimBlock>>>(d_pixelConnections, sourceImage.getGPUAddressOfDimensionsData(),
                                            d_pixelDirections);
    cudaDeviceSynchronize();
}