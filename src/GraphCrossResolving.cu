#include "GraphCrossResolving.hpp"

__device__ void GraphCrossResolving::doAtomicAnd(Graph::byte* address, Graph::byte value)
{
    unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(std::size_t) address & 3];
    unsigned int old, assumed, min_, new_;

    old = *base_address;
    do
    {
        assumed = old;
        min_ = value & (Graph::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
        new_ = __byte_perm(old, min_, sel);
        old = atomicCAS(base_address, assumed, new_);
    }
    while(assumed != old);
}

__device__ void GraphCrossResolving::doAtomicOr(Graph::byte* address, Graph::byte value)
{
    unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
    unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
    unsigned int sel = selectors[(std::size_t) address & 3];
    unsigned int old, assumed, min_, new_;

    old = *base_address;
    do
    {
        assumed = old;
        min_ = value | (Graph::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
        new_ = __byte_perm(old, min_, sel);
        old = atomicCAS(base_address, assumed, new_);
    }
    while(assumed != old);
}

__device__ void GraphCrossResolving::addEdgeConnection(Graph::byte& nodeEdges, GraphEdge direction)
{
    GraphCrossResolving::doAtomicOr(&nodeEdges, static_cast<Graph::byte>(direction));
}

__device__ void GraphCrossResolving::removeEdgeConnection(Graph::byte& nodeEdges, GraphEdge direction)
{
    GraphCrossResolving::doAtomicAnd(&nodeEdges, ~static_cast<Graph::byte>(direction));
}

__device__ bool GraphCrossResolving::isThereAnEdge(const Graph::byte& nodeEdges, GraphEdge direction)
{
    return nodeEdges & static_cast<Graph::byte>(direction);
}

__device__ int GraphCrossResolving::getNodeDegree(const Graph::byte& nodeEdges)
{
    int nodeDegree = 0;

    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::UP);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::DOWN);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::LEFT);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::RIGHT);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::UPPER_LEFT);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::UPPER_RIGHT);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::LOWER_LEFT);
    nodeDegree += GraphCrossResolving::isThereAnEdge(nodeEdges, GraphEdge::LOWER_RIGHT);

    return nodeDegree;
}

__device__ bool GraphCrossResolving::isThereAnIslandNode(const Graph::byte& nodeAEdges, const Graph::byte& nodeBEdges)
{
    return (GraphCrossResolving::getNodeDegree(nodeAEdges) == 1) ||
           (GraphCrossResolving::getNodeDegree(nodeBEdges) == 1);
}

__device__
int GraphCrossResolving::getNeighborRowIdx(int row, GraphEdge direction)
{
    if(direction == GraphEdge::UP || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::UPPER_RIGHT)
        row--;
    if(direction == GraphEdge::DOWN || direction == GraphEdge::LOWER_LEFT || direction == GraphEdge::LOWER_RIGHT)
        row++;

    return row;
}

__device__
int GraphCrossResolving::getNeighborColIdx(int col, GraphEdge direction)
{
    if(direction == GraphEdge::LEFT || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::LOWER_LEFT)
        col--;
    if(direction == GraphEdge::RIGHT || direction == GraphEdge::UPPER_RIGHT || direction == GraphEdge::LOWER_RIGHT)
        col++;

    return col;
}

__device__ GraphEdge GraphCrossResolving::getNeighborInDirectionOtherThanGiven(const Graph::byte& nodeEdges,
                                                                               const GraphEdge forbiddenDirection)
{
    if(GraphCrossResolving::getNodeDegree(nodeEdges) < 2)
        return forbiddenDirection;

    GraphEdge chosenDirection = GraphEdge::UPPER_LEFT;
    do
    {
        if((chosenDirection != forbiddenDirection) && GraphCrossResolving::isThereAnEdge(nodeEdges, chosenDirection))
            return chosenDirection;
        chosenDirection = static_cast<GraphEdge>(static_cast<Graph::byte>(chosenDirection) << 1);
    } while( static_cast<Graph::byte>(chosenDirection) );

    return forbiddenDirection;
}

__device__ GraphEdge GraphCrossResolving::getOppositeDirection(GraphEdge direction)
{
    GraphEdge result;
    switch(direction)
    {
        case GraphEdge::DOWN:        result = GraphEdge::UP; break;
        case GraphEdge::UP:          result = GraphEdge::DOWN; break;
        case GraphEdge::LEFT:        result = GraphEdge::RIGHT; break;
        case GraphEdge::RIGHT:       result = GraphEdge::LEFT; break;
        case GraphEdge::UPPER_RIGHT: result = GraphEdge::LOWER_LEFT; break;
        case GraphEdge::UPPER_LEFT:  result = GraphEdge::LOWER_RIGHT; break;
        case GraphEdge::LOWER_LEFT:  result = GraphEdge::UPPER_RIGHT; break;
        case GraphEdge::LOWER_RIGHT: result = GraphEdge::UPPER_LEFT; break;
    }
    return result;
}

__device__ int GraphCrossResolving::getLengthOfPathComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                                             Graph::byte* edges, int width, int height)
{
    int secondaryNodeRow = GraphCrossResolving::getNeighborRowIdx(row, secondaryNodeDirection);
    int secondaryNodeCol = GraphCrossResolving::getNeighborColIdx(col, secondaryNodeDirection);
    GraphEdge previousDirection = secondaryNodeDirection;

    int result = 1;
    bool wasSecondaryNodeVisited = false;

    int currentIdx = col + row * width;
    while( !wasSecondaryNodeVisited && GraphCrossResolving::getNodeDegree(edges[currentIdx]) == 2 )
    {
        GraphEdge directionOfNextNode = GraphCrossResolving::getNeighborInDirectionOtherThanGiven(
                edges[currentIdx],
                previousDirection);

        previousDirection = GraphCrossResolving::getOppositeDirection(directionOfNextNode);
        row = GraphCrossResolving::getNeighborRowIdx(row, directionOfNextNode);
        col = GraphCrossResolving::getNeighborColIdx(col, directionOfNextNode);

        wasSecondaryNodeVisited = (row == secondaryNodeRow) && (col == secondaryNodeCol);

        currentIdx = col + row * width;
        ++result;
    }

    if(!wasSecondaryNodeVisited)
    {
        currentIdx = secondaryNodeCol + secondaryNodeRow * width;
        previousDirection = GraphCrossResolving::getOppositeDirection(secondaryNodeDirection);
        row = secondaryNodeRow;
        col = secondaryNodeCol;

        while(getNodeDegree(edges[currentIdx]) == 2)
        {
            GraphEdge directionOfNextNode = GraphCrossResolving::getNeighborInDirectionOtherThanGiven(
                    edges[currentIdx],
                    previousDirection);
            previousDirection = GraphCrossResolving::getOppositeDirection(directionOfNextNode);
            row = GraphCrossResolving::getNeighborRowIdx(row, directionOfNextNode);
            col = GraphCrossResolving::getNeighborColIdx(col, directionOfNextNode);

            currentIdx = col + row * width;
            ++result;
        }
    }
    return result;
}

__device__ int GraphCrossResolving::getSizeOfConnectedComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                                                const int radius, const int* labelData,
                                                                int width, int height)
{
    int result = 0;
    int idx = col + row * width;
    int labelDataOfCurrentNode = labelData[idx];

    int rowOfDiagonalNeighbor = getNeighborRowIdx(row, secondaryNodeDirection);
    int colOfDiagonalNeighbor = getNeighborColIdx(col, secondaryNodeDirection);
    row = min(row, rowOfDiagonalNeighbor);
    col = min(col, colOfDiagonalNeighbor);

    for(int rowDiff = -radius; rowDiff <= radius+1; ++rowDiff)
    for(int colDiff = -radius; colDiff <= radius+1; ++colDiff)
    {
        int currentRow = row + rowDiff;
        int currentCol = col + colDiff;
        if(currentCol >= 0 && currentCol < width && currentRow >=0 && currentRow < height)
        {
            result += (labelDataOfCurrentNode == labelData[currentCol + currentRow * width]);
        }
    }

    return result;
}

__global__ void GraphCrossResolving::resolveCriticalCrossings(Graph::byte* edges, const int* labelData, int width,
                                                              int height, const int islandHeuristicMultiplier,
                                                              const int curveHeuristicMultiplier,
                                                              const int sparsePixelsMultiplier, const int sparsePixelsRadius)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height - 1 && col < width - 1)
    {
        int idx = col + row * width;
        Graph::byte edgeValuesOfCurrentNode = edges[idx];

        bool isThereACrossing = GraphCrossResolving::isThereAnEdge(edgeValuesOfCurrentNode, GraphEdge::LOWER_RIGHT) &&
                                GraphCrossResolving::isThereAnEdge(edges[idx + 1], GraphEdge::LOWER_LEFT);
        if(isThereACrossing)
        {
            int weightOfDiagonalLeftURightD = 0;
            int weightOfDiagonalLeftDRightU = 0;

            //island heuristic
            weightOfDiagonalLeftURightD +=
                GraphCrossResolving::isThereAnIslandNode(edgeValuesOfCurrentNode,
                                                         edges[idx + width + 1]) * islandHeuristicMultiplier;
            weightOfDiagonalLeftDRightU +=
                GraphCrossResolving::isThereAnIslandNode(edges[idx + width],
                                                         edges[idx + 1]) * islandHeuristicMultiplier;

            //curve heuristic
            int lengthOfLeftURightDCurve =
                    GraphCrossResolving::getLengthOfPathComponent(row,col,GraphEdge::LOWER_RIGHT, edges, width, height);
            int lengthOfLeftDRightUCurve =
                    GraphCrossResolving::getLengthOfPathComponent(row+1,col,GraphEdge::UPPER_RIGHT, edges, width, height);

            if(lengthOfLeftURightDCurve > lengthOfLeftDRightUCurve)
                weightOfDiagonalLeftURightD +=
                        (lengthOfLeftURightDCurve - lengthOfLeftDRightUCurve) * curveHeuristicMultiplier;
            else
                weightOfDiagonalLeftDRightU +=
                        (lengthOfLeftDRightUCurve - lengthOfLeftURightDCurve) * curveHeuristicMultiplier;

            //sparse pixels heuristic
            int sizeOfLeftURightDComponent = GraphCrossResolving::getSizeOfConnectedComponent(row, col,
                                                                                              GraphEdge::LOWER_RIGHT,
                                                                                              sparsePixelsRadius,
                                                                                              labelData, width, height);
            int sizeOfLeftDRightUComponent = GraphCrossResolving::getSizeOfConnectedComponent(row+1, col,
                                                                                              GraphEdge::UPPER_RIGHT,
                                                                                              sparsePixelsRadius,
                                                                                              labelData, width, height);
            if(sizeOfLeftURightDComponent > sizeOfLeftDRightUComponent)
                weightOfDiagonalLeftDRightU +=
                        (sizeOfLeftURightDComponent - sizeOfLeftDRightUComponent) * sparsePixelsMultiplier;
            else
                weightOfDiagonalLeftURightD +=
                        (sizeOfLeftDRightUComponent - sizeOfLeftURightDComponent) * sparsePixelsMultiplier;


            if(weightOfDiagonalLeftURightD > weightOfDiagonalLeftDRightU)
            {
                removeEdgeConnection(edges[idx + 1], GraphEdge::LOWER_LEFT);
                removeEdgeConnection(edges[idx + width], GraphEdge::UPPER_RIGHT);
            }
            else if(weightOfDiagonalLeftURightD < weightOfDiagonalLeftDRightU)
            {
                removeEdgeConnection(edges[idx], GraphEdge::LOWER_RIGHT);
                removeEdgeConnection(edges[idx + width + 1], GraphEdge::UPPER_LEFT);
            }
            else
            {
                removeEdgeConnection(edges[idx + 1], GraphEdge::LOWER_LEFT);
                removeEdgeConnection(edges[idx + width], GraphEdge::UPPER_RIGHT);
                removeEdgeConnection(edges[idx], GraphEdge::LOWER_RIGHT);
                removeEdgeConnection(edges[idx + width + 1], GraphEdge::UPPER_LEFT);
            }
        }
    }
}