#include "PolygonSideMapConstructing.hpp"

__device__ bool PolygonSideMapConstructing::isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction)
{
    return nodeEdges & static_cast<Graph::byte>(direction);
}

__device__ bool
PolygonSideMapConstructing::isThisQuarterForwardSlash(int row, int col, int width, const Graph::byte* graphData)
{
    int idxOfGraphEntry = col + row * width;
    Graph::byte checkedGraphData = graphData[idxOfGraphEntry];
    return isThereAnEdge(checkedGraphData, GraphEdge::UPPER_LEFT);
}

__device__ bool
PolygonSideMapConstructing::isThisQuarterBackslash(int row, int col, int width, const Graph::byte* graphData)
{
    int idxOfGraphEntry = col - 1 + row * width;
    Graph::byte checkedGraphData = graphData[idxOfGraphEntry];
    return isThereAnEdge(checkedGraphData, GraphEdge::UPPER_RIGHT);
}

__global__ void
PolygonSideMapConstructing::createPolygonSide(PolygonSide* sideData, const Graph::byte* graphData, int width, int height)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height && col < width)
    {
        int idxOfQuarter = col + row * width;
        PolygonSide::Type currentQuarterSideType = PolygonSide::Type::Point;
        if(row != 0 && col != 0)
        {
            if(isThisQuarterBackslash(row, col, width, graphData))
                currentQuarterSideType = PolygonSide::Type::Backslash;
            else if(isThisQuarterForwardSlash(row, col, width, graphData))
                currentQuarterSideType = PolygonSide::Type::ForwardSlash;
        }

        PolygonSide::point_type rowA = static_cast<PolygonSide::point_type>(row * 100);
        PolygonSide::point_type colA = static_cast<PolygonSide::point_type>(col * 100);
        PolygonSide::point_type rowB = static_cast<PolygonSide::point_type>(row * 100);
        PolygonSide::point_type colB = static_cast<PolygonSide::point_type>(col * 100);
        if(currentQuarterSideType == PolygonSide::Type::ForwardSlash)
        {
            rowA -= 25.0f;
            colA += 25.0f;
            rowB += 25.0f;
            colB -= 25.0f;
        }
        else if(currentQuarterSideType == PolygonSide::Type::Backslash)
        {
            rowA -= 25.0f;
            colA -= 25.0f;
            rowB += 25.0f;
            colB += 25.0f;
        }

        sideData[idxOfQuarter].info = static_cast<PolygonSide::info_type>(currentQuarterSideType);
        sideData[idxOfQuarter].pointA[0] = rowA;
        sideData[idxOfQuarter].pointA[1] = colA;
        sideData[idxOfQuarter].pointB[0] = rowB;
        sideData[idxOfQuarter].pointB[1] = colB;
    }
}

void PolygonSideMapConstructing::createMap(PolygonSide* sideData, const Graph::byte* graphData, int width, int height)
{
    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x, (width + dimBlock.y -1)/dimBlock.y);
    PolygonSideMapConstructing::createPolygonSide<<<dimGrid, dimBlock>>>(sideData, graphData, width, height);
    cudaDeviceSynchronize();
}

void PolygonSideMapConstructing::getCreatedMapData(std::vector<PolygonSide>& output, PolygonSide* d_sideData, int width,
                                                   int height)
{
    PolygonSide* cellSideValues = new PolygonSide[width * height];
    cudaMemcpy(cellSideValues, d_sideData, width * height * sizeof(PolygonSide), cudaMemcpyDeviceToHost);
    output = std::vector<PolygonSide>{cellSideValues, cellSideValues + width * height};
    delete[] cellSideValues;
}