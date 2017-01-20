#ifndef VCTRSKL_CROSSINGRESOLVING_HPP
#define VCTRSKL_CROSSINGRESOLVING_HPP

#include "Constants.hpp"

namespace CrossingResolving
{
    __device__ void doAtomicAnd(Graph::byte* address, Graph::byte value)
    {
        unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t) address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do
        {
            assumed = old;
            min_ = value & (Color::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        }
        while(assumed != old);
    }

    __device__ void doAtomicOr(Graph::byte* address, Graph::byte value)
    {
        unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t) address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do
        {
            assumed = old;
            min_ = value | (Color::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        }
        while(assumed != old);
    }

    __device__ void addEdgeConnection(PixelGraph::edge_type& nodeEdges, GraphEdge direction)
    {
        CrossingResolving::doAtomicOr(&nodeEdges, static_cast<PixelGraph::edge_type>(direction));
    }

    __device__ void removeEdgeConnection(PixelGraph::edge_type& nodeEdges, GraphEdge direction)
    {
        CrossingResolving::doAtomicAnd(&nodeEdges, ~static_cast<PixelGraph::edge_type>(direction));
    }

    __device__
    int getNeighborRowIdx(int row, GraphEdge direction)
    {
        if(direction == GraphEdge::UP || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::UPPER_RIGHT)
            row--;
        if(direction == GraphEdge::DOWN || direction == GraphEdge::LOWER_LEFT || direction == GraphEdge::LOWER_RIGHT)
            row++;

        return row;
    }

    __device__
    int getNeighborColIdx(int col, GraphEdge direction)
    {
        if(direction == GraphEdge::LEFT || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::LOWER_LEFT)
            col--;
        if(direction == GraphEdge::RIGHT || direction == GraphEdge::UPPER_RIGHT || direction == GraphEdge::LOWER_RIGHT)
            col++;

        return col;
    }

    __global__ void removeUnnecessaryCrossings(PixelGraph::edge_type* edges, const std::size_t* dim)
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
                addEdgeConnection(edges[idx], GraphEdge::RIGHT);
                addEdgeConnection(edges[idx], GraphEdge::DOWN);
                addEdgeConnection(edges[idx + 1], GraphEdge::DOWN);
                addEdgeConnection(edges[idx + 1], GraphEdge::LEFT);
                addEdgeConnection(edges[idx + dim[1]], GraphEdge::RIGHT);
                addEdgeConnection(edges[idx + dim[1]], GraphEdge::UP);
                addEdgeConnection(edges[idx + dim[1] + 1], GraphEdge::LEFT);
                addEdgeConnection(edges[idx + dim[1] + 1], GraphEdge::UP);

                removeEdgeConnection(edges[idx + 1], GraphEdge::LOWER_LEFT);
                removeEdgeConnection(edges[idx + dim[1]], GraphEdge::UPPER_RIGHT);
                removeEdgeConnection(edges[idx], GraphEdge::LOWER_RIGHT);
                removeEdgeConnection(edges[idx + dim[1] + 1], GraphEdge::UPPER_LEFT);
            }
        }
    }

    __device__ bool isThereAnEdge(const PixelGraph::edge_type& nodeEdges, GraphEdge direction)
    {
        return nodeEdges & static_cast<PixelGraph::edge_type>(direction);
    }

    __device__ int getNodeDegree(const PixelGraph::edge_type& nodeEdges)
    {
        int nodeDegree = 0;

        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::UP);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::DOWN);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::LEFT);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::RIGHT);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::UPPER_LEFT);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::UPPER_RIGHT);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::LOWER_LEFT);
        nodeDegree += isThereAnEdge(nodeEdges, GraphEdge::LOWER_RIGHT);

        return nodeDegree;
    }

    __device__ bool isThereAnIslandNode(const PixelGraph::edge_type& nodeAEdges, const PixelGraph::edge_type& nodeBEdges)
    {
        return (getNodeDegree(nodeAEdges) == 1) || (getNodeDegree(nodeBEdges) == 1);
    }

    __device__ GraphEdge getOppositeDirection(GraphEdge direction)
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

    __device__ GraphEdge getNeighborInDirectionOtherThanGiven(const PixelGraph::edge_type& nodeEdges,
                                                              const GraphEdge forbiddenDirection)
    {
        if(getNodeDegree(nodeEdges) < 2)
            return forbiddenDirection;

        GraphEdge chosenDirection = GraphEdge::UPPER_LEFT;
        do
        {
            if((chosenDirection != forbiddenDirection) && isThereAnEdge(nodeEdges, chosenDirection))
                return chosenDirection;
            chosenDirection = static_cast<GraphEdge>(static_cast<PixelGraph::edge_type>(chosenDirection) << 1);
        } while( static_cast<PixelGraph::edge_type>(chosenDirection) );

        return forbiddenDirection;
    }

    __device__ int getLengthOfPathComponent(int row, int col, GraphEdge secondaryNodeDirection, const std::size_t* dim,
                                            PixelGraph::edge_type* edges)
    {
        int secondaryNodeRow = getNeighborRowIdx(row, secondaryNodeDirection);
        int secondaryNodeCol = getNeighborColIdx(col, secondaryNodeDirection);
        GraphEdge previousDirection = secondaryNodeDirection;

        int result = 1;
        bool wasSecondaryNodeVisited = false;

        std::size_t currentIdx = col + row * dim[1];
        while( !wasSecondaryNodeVisited && getNodeDegree(edges[currentIdx]) == 2 )
        {
            GraphEdge directionOfNextNode = getNeighborInDirectionOtherThanGiven(edges[currentIdx], previousDirection);

            previousDirection = getOppositeDirection(directionOfNextNode);
            row = getNeighborRowIdx(row, directionOfNextNode);
            col = getNeighborColIdx(col, directionOfNextNode);

            wasSecondaryNodeVisited = (row == secondaryNodeRow) && (col == secondaryNodeCol);

            currentIdx = col + row * dim[1];
            ++result;
        }

        if(!wasSecondaryNodeVisited)
        {
            currentIdx = secondaryNodeCol + secondaryNodeRow * dim[1];
            previousDirection = getOppositeDirection(secondaryNodeDirection);
            row = secondaryNodeRow;
            col = secondaryNodeCol;

            while(getNodeDegree(edges[currentIdx]) == 2)
            {
                GraphEdge directionOfNextNode = getNeighborInDirectionOtherThanGiven(edges[currentIdx],
                                                                                     previousDirection);
                previousDirection = getOppositeDirection(directionOfNextNode);
                row = getNeighborRowIdx(row, directionOfNextNode);
                col = getNeighborColIdx(col, directionOfNextNode);

                currentIdx = col + row * dim[1];
                ++result;
            }
        }
        return result;
    }

    __global__ void resolveCriticalCrossings(PixelGraph::edge_type* edges, const std::size_t* dim)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        if(i < dim[0] - 1 && j < dim[1] - 1)
        {
            std::size_t idx = j + i * dim[1];
            bool isThereACrossing = isThereAnEdge(edges[idx], GraphEdge::LOWER_RIGHT) &&
                                    isThereAnEdge(edges[idx + 1], GraphEdge::LOWER_LEFT);
            if(isThereACrossing)
            {
                int weightOfDiagonalLeftURightD = 0;
                int weightOfDiagonalLeftDRightU = 0;

                //TODO make heuristic multipliers externally configurable
                const int islandHeuristicMultiplier = 5;
                const int curveHeuristicMultiplier = 1;

                weightOfDiagonalLeftURightD +=
                        isThereAnIslandNode(edges[idx], edges[idx + dim[1] + 1]) * islandHeuristicMultiplier;
                weightOfDiagonalLeftDRightU +=
                        isThereAnIslandNode(edges[idx + dim[1]], edges[idx + 1]) * islandHeuristicMultiplier;

                weightOfDiagonalLeftURightD +=
                        getLengthOfPathComponent(i,j,GraphEdge::LOWER_RIGHT, dim, edges) * curveHeuristicMultiplier;
                weightOfDiagonalLeftDRightU +=
                        getLengthOfPathComponent(i+1,j,GraphEdge::UPPER_RIGHT, dim, edges) * curveHeuristicMultiplier;

                if(weightOfDiagonalLeftURightD > weightOfDiagonalLeftDRightU)
                {
                    removeEdgeConnection(edges[idx + 1], GraphEdge::LOWER_LEFT);
                    removeEdgeConnection(edges[idx + dim[1]], GraphEdge::UPPER_RIGHT);
                }
                else if(weightOfDiagonalLeftURightD < weightOfDiagonalLeftDRightU)
                {
                    removeEdgeConnection(edges[idx], GraphEdge::LOWER_RIGHT);
                    removeEdgeConnection(edges[idx + dim[1] + 1], GraphEdge::UPPER_LEFT);
                }
                else
                {
                    removeEdgeConnection(edges[idx + 1], GraphEdge::LOWER_LEFT);
                    removeEdgeConnection(edges[idx + dim[1]], GraphEdge::UPPER_RIGHT);
                    removeEdgeConnection(edges[idx], GraphEdge::LOWER_RIGHT);
                    removeEdgeConnection(edges[idx + dim[1] + 1], GraphEdge::UPPER_LEFT);
                }
            }
        }
    }
}

#endif //VCTRSKL_CROSSINGRESOLVING_HPP
