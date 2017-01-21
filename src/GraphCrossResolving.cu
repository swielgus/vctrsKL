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

__global__ void GraphCrossResolving::removeUnnecessaryCrossings(PixelGraphInfo* graphInfo)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < graphInfo->height - 1 && col < graphInfo->width - 1)
    {
        std::size_t idx = col + row * graphInfo->width;

        Graph::byte upperLeftConnected[12]  = {255, 48,  24,  255, 56,  255, 255, 255, 255, 40,  40,  255};
        Graph::byte lowerLeftConnected[12]  = {96,  255, 255, 192, 255, 255, 224, 255, 255, 160, 255, 160};
        Graph::byte upperRightConnected[12] = {6,   255, 255, 12,  255, 14,  255, 255, 10,  255, 10,  255};
        Graph::byte lowerRightConnected[12] = {255, 3,   129, 255, 255, 255, 255, 131, 130, 255, 255, 130};

        int k = 0;
        bool squareIsNotConnected = true;
        while(k < 12 && squareIsNotConnected)
        {
            squareIsNotConnected = !(
                ((upperLeftConnected[k] == 255) || ((upperLeftConnected[k] & graphInfo->edges[idx]) == upperLeftConnected[k])) &&
                ((upperRightConnected[k] == 255) || ((upperRightConnected[k] & graphInfo->edges[idx + 1]) == upperRightConnected[k])) &&
                ((lowerLeftConnected[k] == 255) || ((lowerLeftConnected[k] & graphInfo->edges[idx + graphInfo->width]) == lowerLeftConnected[k])) &&
                ((lowerRightConnected[k] == 255) || ((lowerRightConnected[k] & graphInfo->edges[idx + graphInfo->width + 1]) == lowerRightConnected[k]))
            );
            ++k;
        }

        if(!squareIsNotConnected)
        {
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx], GraphEdge::RIGHT);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx], GraphEdge::DOWN);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + 1], GraphEdge::DOWN);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + 1], GraphEdge::LEFT);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + graphInfo->width], GraphEdge::RIGHT);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + graphInfo->width], GraphEdge::UP);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + graphInfo->width + 1], GraphEdge::LEFT);
            GraphCrossResolving::addEdgeConnection(graphInfo->edges[idx + graphInfo->width + 1], GraphEdge::UP);

            GraphCrossResolving::removeEdgeConnection(graphInfo->edges[idx + 1], GraphEdge::LOWER_LEFT);
            GraphCrossResolving::removeEdgeConnection(graphInfo->edges[idx + graphInfo->width], GraphEdge::UPPER_RIGHT);
            GraphCrossResolving::removeEdgeConnection(graphInfo->edges[idx], GraphEdge::LOWER_RIGHT);
            GraphCrossResolving::removeEdgeConnection(graphInfo->edges[idx + graphInfo->width + 1], GraphEdge::UPPER_LEFT);
        }
    }
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
                                                             PixelGraphInfo* graphInfo)
{
    int secondaryNodeRow = GraphCrossResolving::getNeighborRowIdx(row, secondaryNodeDirection);
    int secondaryNodeCol = GraphCrossResolving::getNeighborColIdx(col, secondaryNodeDirection);
    GraphEdge previousDirection = secondaryNodeDirection;

    int result = 1;
    bool wasSecondaryNodeVisited = false;

    std::size_t currentIdx = col + row * graphInfo->width;
    while( !wasSecondaryNodeVisited && GraphCrossResolving::getNodeDegree(graphInfo->edges[currentIdx]) == 2 )
    {
        GraphEdge directionOfNextNode = GraphCrossResolving::getNeighborInDirectionOtherThanGiven(
                graphInfo->edges[currentIdx],
                previousDirection);

        previousDirection = GraphCrossResolving::getOppositeDirection(directionOfNextNode);
        row = GraphCrossResolving::getNeighborRowIdx(row, directionOfNextNode);
        col = GraphCrossResolving::getNeighborColIdx(col, directionOfNextNode);

        wasSecondaryNodeVisited = (row == secondaryNodeRow) && (col == secondaryNodeCol);

        currentIdx = col + row * graphInfo->width;
        ++result;
    }

    if(!wasSecondaryNodeVisited)
    {
        currentIdx = secondaryNodeCol + secondaryNodeRow * graphInfo->width;
        previousDirection = GraphCrossResolving::getOppositeDirection(secondaryNodeDirection);
        row = secondaryNodeRow;
        col = secondaryNodeCol;

        while(getNodeDegree(graphInfo->edges[currentIdx]) == 2)
        {
            GraphEdge directionOfNextNode = GraphCrossResolving::getNeighborInDirectionOtherThanGiven(
                    graphInfo->edges[currentIdx],
                    previousDirection);
            previousDirection = GraphCrossResolving::getOppositeDirection(directionOfNextNode);
            row = GraphCrossResolving::getNeighborRowIdx(row, directionOfNextNode);
            col = GraphCrossResolving::getNeighborColIdx(col, directionOfNextNode);

            currentIdx = col + row * graphInfo->width;
            ++result;
        }
    }
    return result;
}

__device__ int GraphCrossResolving::getSizeOfConnectedComponent(int row, int col, GraphEdge secondaryNodeDirection,
                                                                const std::size_t& radius, PixelGraphInfo* graphInfo)
{
    const std::size_t checkedRectangleSide = 2 * radius + 1;
    bool* wasNodeVisited = new bool[checkedRectangleSide * checkedRectangleSide];
    //cudaMalloc( &wasNodeVisited, checkedRectangleSide * checkedRectangleSide * sizeof(bool));
    for(int i = 0; i < checkedRectangleSide * checkedRectangleSide; ++i)
        wasNodeVisited[i] = false;

    wasNodeVisited[radius + radius * checkedRectangleSide] = true;
    int result = 1;




    delete[] wasNodeVisited;
    //cudaFree(wasNodeVisited);
    return result;
}

__global__ void GraphCrossResolving::resolveCriticalCrossings(PixelGraphInfo* graphInfo)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < graphInfo->height - 1 && col < graphInfo->width - 1)
    {
        std::size_t idx = col + row * graphInfo->width;
        bool isThereACrossing = GraphCrossResolving::isThereAnEdge(graphInfo->edges[idx], GraphEdge::LOWER_RIGHT) &&
                                GraphCrossResolving::isThereAnEdge(graphInfo->edges[idx + 1], GraphEdge::LOWER_LEFT);
        if(isThereACrossing)
        {
            int weightOfDiagonalLeftURightD = 0;
            int weightOfDiagonalLeftDRightU = 0;

            //TODO make heuristic multipliers externally configurable
            const int islandHeuristicMultiplier = 5;
            const int curveHeuristicMultiplier = 1;
            const int sparsePixelsMultiplier = 1;
            const int sparsePixelsRadius = 3;

            //island heuristic
            weightOfDiagonalLeftURightD +=
                GraphCrossResolving::isThereAnIslandNode(graphInfo->edges[idx],
                                                         graphInfo->edges[idx + graphInfo->width + 1]) * islandHeuristicMultiplier;
            weightOfDiagonalLeftDRightU +=
                GraphCrossResolving::isThereAnIslandNode(graphInfo->edges[idx + graphInfo->width],
                                                         graphInfo->edges[idx + 1]) * islandHeuristicMultiplier;

            //curve heuristic
            int lengthOfLeftURightDCurve =
                    GraphCrossResolving::getLengthOfPathComponent(row,col,GraphEdge::LOWER_RIGHT, graphInfo);
            int lengthOfLeftDRightUCurve =
                    GraphCrossResolving::getLengthOfPathComponent(row+1,col,GraphEdge::UPPER_RIGHT, graphInfo);

            if(lengthOfLeftURightDCurve > lengthOfLeftDRightUCurve)
                weightOfDiagonalLeftURightD +=
                        (lengthOfLeftURightDCurve - lengthOfLeftDRightUCurve) * curveHeuristicMultiplier;
            else
                weightOfDiagonalLeftDRightU +=
                        (lengthOfLeftDRightUCurve - lengthOfLeftURightDCurve) * curveHeuristicMultiplier;

            //sparse pixels heuristic
            int sizeOfLeftURightDComponent = GraphCrossResolving::getSizeOfConnectedComponent(row, col,
                                                                                              GraphEdge::LOWER_RIGHT,
                                                                                              sparsePixelsRadius, graphInfo);
            int sizeOfLeftDRightUComponent = GraphCrossResolving::getSizeOfConnectedComponent(row+1, col,
                                                                                              GraphEdge::UPPER_RIGHT,
                                                                                              sparsePixelsRadius, graphInfo);
            if(sizeOfLeftURightDComponent > sizeOfLeftDRightUComponent)
                weightOfDiagonalLeftDRightU +=
                        (sizeOfLeftURightDComponent - sizeOfLeftDRightUComponent) * sparsePixelsMultiplier;
            else
                weightOfDiagonalLeftURightD +=
                        (sizeOfLeftDRightUComponent - sizeOfLeftURightDComponent) * sparsePixelsMultiplier;


            if(weightOfDiagonalLeftURightD > weightOfDiagonalLeftDRightU)
            {
                removeEdgeConnection(graphInfo->edges[idx + 1], GraphEdge::LOWER_LEFT);
                removeEdgeConnection(graphInfo->edges[idx + graphInfo->width], GraphEdge::UPPER_RIGHT);
            }
            else if(weightOfDiagonalLeftURightD < weightOfDiagonalLeftDRightU)
            {
                removeEdgeConnection(graphInfo->edges[idx], GraphEdge::LOWER_RIGHT);
                removeEdgeConnection(graphInfo->edges[idx + graphInfo->width + 1], GraphEdge::UPPER_LEFT);
            }
            else
            {
                removeEdgeConnection(graphInfo->edges[idx + 1], GraphEdge::LOWER_LEFT);
                removeEdgeConnection(graphInfo->edges[idx + graphInfo->width], GraphEdge::UPPER_RIGHT);
                removeEdgeConnection(graphInfo->edges[idx], GraphEdge::LOWER_RIGHT);
                removeEdgeConnection(graphInfo->edges[idx + graphInfo->width + 1], GraphEdge::UPPER_LEFT);
            }
        }
    }
}