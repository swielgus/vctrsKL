#include "PixelGraph.hpp"
#include "GraphCrossResolving.hpp"
#include "GraphConstructing.hpp"

PixelGraph::PixelGraph(const ImageData& image)
    : sourceImage{image}, d_pixelConnections{nullptr}, d_graphInfo{nullptr}
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
    cudaFree(d_graphInfo);
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

    PixelGraphInfo graphInfo{d_pixelConnections, width, height};
    cudaMalloc( &d_graphInfo, sizeof(PixelGraphInfo));
    cudaMemcpy(d_graphInfo, &graphInfo, sizeof(PixelGraphInfo), cudaMemcpyHostToDevice);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    //cudaEventRecord(start);
    GraphConstructing::createConnections<<<dimGrid, dimBlock>>>(d_graphInfo, sourceImage.getGPUAddressOfYColorData(),
                                                                sourceImage.getGPUAddressOfUColorData(),
                                                                sourceImage.getGPUAddressOfVColorData());
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

    std::vector< std::vector<edge_type> > result;
    result.resize(height);
    for(std::vector<edge_type>& row : result)
        row.resize(width);

    edge_type* pixelDirection = new edge_type[width * height];
    cudaMemcpy(pixelDirection, d_pixelConnections, width * height * sizeof(edge_type), cudaMemcpyDeviceToHost);

    for(std::size_t x = 0; x < height; ++x)
    for(std::size_t y = 0; y < width; ++y)
    {
        result[x][y] = *(pixelDirection + (y + x * width));
    }

    delete[] pixelDirection;

    return result;
}

void PixelGraph::resolveCrossings()
{
    const std::size_t width = sourceImage.getWidth();
    const std::size_t height = sourceImage.getHeight();
    dim3 dimBlock(32, 32);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    GraphCrossResolving::removeUnnecessaryCrossings<<<dimGrid, dimBlock>>>(d_graphInfo);
    cudaDeviceSynchronize();

    GraphCrossResolving::resolveCriticalCrossings<<<dimGrid, dimBlock>>>(d_graphInfo);
    cudaDeviceSynchronize();
}