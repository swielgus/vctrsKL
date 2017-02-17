#include "PixelGraph.hpp"
#include "GraphCrossResolving.hpp"
#include "GraphConstructing.hpp"

PixelGraph::PixelGraph(const ImageData& image, int islandMultiplier, int curveMultiplier,
                       int pixelsMultiplier, int pixelsRadius)
    : sourceImage{image}, d_pixelConnections{nullptr}, islandHeuristicMultiplier{islandMultiplier},
      curveHeuristicMultiplier{curveMultiplier}, sparsePixelsMultiplier{pixelsMultiplier},
      sparsePixelsRadius{pixelsRadius}
{
    constructGraph();
}

PixelGraph::~PixelGraph()
{
    freeDeviceData();
    //cudaDeviceReset();
}

const PixelGraph::edge_type* PixelGraph::getGPUAddressOfGraphData() const
{
    return d_pixelConnections;
}

std::size_t PixelGraph::getWidth() const
{
    return sourceImage.getWidth();
}

std::size_t PixelGraph::getHeight() const
{
    return sourceImage.getHeight();
}

void PixelGraph::freeDeviceData()
{
    cudaFree(d_pixelConnections);
}

void PixelGraph::constructGraph()
{
    freeDeviceData();

    const std::size_t width = getWidth();
    const std::size_t height = getHeight();
    const int* addressOfLabelData = sourceImage.getGPUAddressOfLabelData();
    cudaMalloc( &d_pixelConnections, width * height * sizeof(edge_type));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);
    GraphConstructing::createConnections<<<dimGrid, dimBlock>>>(d_pixelConnections, sourceImage.getGPUAddressOfLabelData(), width, height);
    cudaDeviceSynchronize();
}

std::vector< std::vector<PixelGraph::edge_type> > PixelGraph::getEdgeValues() const
{
    const std::size_t width = getWidth();
    const std::size_t height = getHeight();

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


std::vector<PixelGraph::edge_type> PixelGraph::get1DEdgeValues() const
{
    const std::size_t width = getWidth();
    const std::size_t height = getHeight();

    std::vector<edge_type> result;
    result.resize(height * width);

    edge_type* pixelDirection = new edge_type[width * height];
    cudaMemcpy(pixelDirection, d_pixelConnections, width * height * sizeof(edge_type), cudaMemcpyDeviceToHost);

    for(std::size_t i = 0; i < height * width; ++i)
        result[i] = *(pixelDirection + i);

    delete[] pixelDirection;

    return result;
}

void PixelGraph::resolveCrossings()
{
    const std::size_t width = getWidth();
    const std::size_t height = getHeight();
    dim3 dimBlock(32, 32);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x, (width + dimBlock.y -1)/dimBlock.y);

    GraphCrossResolving::resolveCriticalCrossings<<<dimGrid, dimBlock>>>(
        d_pixelConnections, sourceImage.getGPUAddressOfLabelData(), width, height, islandHeuristicMultiplier,
        curveHeuristicMultiplier, sparsePixelsMultiplier, sparsePixelsRadius);
    cudaDeviceSynchronize();

    //TODO (opt) update labels after disconnecting components via crossing resolving
}