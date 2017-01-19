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

__device__
int getNeighborRowIdx(int row, GraphEdge direction, const std::size_t* dim)
{
    if(direction == GraphEdge::UP || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::UPPER_RIGHT)
        row--;
    if(direction == GraphEdge::DOWN || direction == GraphEdge::LOWER_LEFT || direction == GraphEdge::LOWER_RIGHT)
        row++;

    return row;
}

__device__
int getNeighborColIdx(int col, GraphEdge direction, const std::size_t* dim)
{
    if(direction == GraphEdge::LEFT || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::LOWER_LEFT)
        col--;
    if(direction == GraphEdge::RIGHT || direction == GraphEdge::UPPER_RIGHT || direction == GraphEdge::LOWER_RIGHT)
        col++;

    return col;
}

__device__ PixelGraph::edge_type
getConnection(int row, int col, GraphEdge direction, const std::size_t* dim, const PixelGraph::color_type* colorY,
              const PixelGraph::color_type* colorU, const PixelGraph::color_type* colorV)
{
    std::size_t idx = col + row * dim[1];

    int neighborRow = getNeighborRowIdx(row, direction, dim);
    int neighborCol = getNeighborColIdx(col, direction, dim);

    PixelGraph::edge_type result = 0;
    if( (neighborRow >= 0 && neighborRow < dim[0]) && (neighborCol >= 0 && neighborCol < dim[1]) )
    {
        std::size_t comparedIdx = neighborCol + neighborRow * dim[1];
        if( areYUVColorsSimilar(colorY[idx], colorU[idx], colorV[idx],
                                colorY[comparedIdx], colorU[comparedIdx], colorV[comparedIdx]))
            result = static_cast<PixelGraph::edge_type>(direction);
    }
    return result;
}

__global__ void
createConnections(PixelGraph::edge_type* edges, const PixelGraph::color_type* colorY,
                  const PixelGraph::color_type* colorU, const PixelGraph::color_type* colorV, const std::size_t* dim)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    if(i < dim[0] && j < dim[1])
    {
        std::size_t idx = j + i * dim[1];
        edges[idx] = 0;

        edges[idx] += getConnection(i, j, GraphEdge::UP, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::DOWN, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::LEFT, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::RIGHT, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::UPPER_RIGHT, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::UPPER_LEFT, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::LOWER_RIGHT, dim, colorY, colorU, colorV);
        edges[idx] += getConnection(i, j, GraphEdge::LOWER_LEFT, dim, colorY, colorU, colorV);
    }
}

PixelGraph::PixelGraph(const ImageData& image)
    : sourceImage{image}, d_pixelConnections{nullptr}
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
}

void PixelGraph::constructGraph()
{
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    freeDeviceData();

    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();
    cudaMalloc( &d_pixelConnections, width * height * sizeof(edge_type));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    //cudaEventRecord(start);
    createConnections<<<dimGrid, dimBlock>>>(d_pixelConnections, sourceImage.getGPUAddressOfYColorData(),
                                    sourceImage.getGPUAddressOfUColorData(), sourceImage.getGPUAddressOfVColorData(),
                                    sourceImage.getGPUAddressOfDimensionsData());
    cudaDeviceSynchronize();
    //cudaEventRecord(stop);

    //cudaEventSynchronize(stop);
    //float milliseconds = 0;
    //cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("time:%f\n", milliseconds);
}

std::vector< std::vector<PixelGraph::edge_type> > PixelGraph::getEdgeValues() const
{
    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();

    std::vector<std::vector<PixelGraph::edge_type>> result;
    result.resize(height);
    for(std::vector<PixelGraph::edge_type>& row : result)
        row.resize(width);

    color_type* pixelDirection = new color_type[width * height];
    cudaMemcpy(pixelDirection, d_pixelConnections, width * height * sizeof(edge_type), cudaMemcpyDeviceToHost);

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
    dim3 dimBlock(32, 32);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    CrossingResolving::removeUnnecessaryCrossings<<<dimGrid, dimBlock>>>(d_pixelConnections,
                                                                         sourceImage.getGPUAddressOfDimensionsData());
    cudaDeviceSynchronize();
}

void PixelGraph::resolveDisconnectingDiagonals()
{
    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();
    dim3 dimBlock(32, 32);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    CrossingResolving::resolveCriticalCrossings<<<dimGrid, dimBlock>>>(d_pixelConnections,
                                                                      sourceImage.getGPUAddressOfDimensionsData());
    cudaDeviceSynchronize();
}