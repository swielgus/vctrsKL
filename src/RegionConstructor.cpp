#include <algorithm>
#include <termcap.h>
#include "RegionConstructor.hpp"

RegionConstructor::RegionConstructor()
    : graphData{}, widthOfGraph{0}, heightOfGraph{0}, createdPointPaths{}, wasNodeVisited{}
{}

RegionConstructor::RegionConstructor(const std::vector<PixelGraph::edge_type>&& usedGraphData, int width, int height)
    : graphData{std::move(usedGraphData)}, widthOfGraph{width}, heightOfGraph{height}, createdPointPaths{}, wasNodeVisited{}
{
    wasNodeVisited.resize(widthOfGraph * heightOfGraph, false);
    runRadialSweeps();
}

RegionConstructor::~RegionConstructor()
{
}


bool RegionConstructor::isThereAnEdge(const PixelGraph::edge_type& currentNodeData, const GraphEdge& direction) const
{
    return currentNodeData & static_cast<PixelGraph::edge_type>(direction);
}

void RegionConstructor::rotateDirectionCounterClockwiseByOne(GraphEdge& direction)
{
    switch(direction)
    {
        case GraphEdge::DOWN:        direction = GraphEdge::LOWER_RIGHT; break;
        case GraphEdge::LOWER_RIGHT: direction = GraphEdge::RIGHT; break;
        case GraphEdge::UPPER_LEFT:  direction = GraphEdge::LEFT; break;
        case GraphEdge::UP:          direction = GraphEdge::UPPER_LEFT; break;
        case GraphEdge::LEFT:        direction = GraphEdge::LOWER_LEFT; break;
        case GraphEdge::LOWER_LEFT:  direction = GraphEdge::DOWN; break;
        case GraphEdge::UPPER_RIGHT: direction = GraphEdge::UP; break;
        case GraphEdge::RIGHT:       direction = GraphEdge::UPPER_RIGHT; break;
    }
}

void RegionConstructor::rotateDirectionCounterClockwiseByTwo(GraphEdge& direction)
{
    switch(direction)
    {
        case GraphEdge::DOWN:        direction = GraphEdge::RIGHT; break;
        case GraphEdge::LOWER_RIGHT: direction = GraphEdge::UPPER_RIGHT; break;
        case GraphEdge::UPPER_LEFT:  direction = GraphEdge::LOWER_LEFT; break;
        case GraphEdge::UP:          direction = GraphEdge::LEFT; break;
        case GraphEdge::LEFT:        direction = GraphEdge::DOWN; break;
        case GraphEdge::LOWER_LEFT:  direction = GraphEdge::LOWER_RIGHT; break;
        case GraphEdge::UPPER_RIGHT: direction = GraphEdge::UPPER_LEFT; break;
        case GraphEdge::RIGHT:       direction = GraphEdge::UP; break;
    }
}

GraphEdge RegionConstructor::getInvertedDirection(const GraphEdge& direction) const
{
    GraphEdge result = direction;
    switch(direction)
    {
        case GraphEdge::DOWN:        result = GraphEdge::UP; break;
        case GraphEdge::LOWER_RIGHT: result = GraphEdge::UPPER_LEFT; break;
        case GraphEdge::UPPER_LEFT:  result = GraphEdge::LOWER_RIGHT; break;
        case GraphEdge::UP:          result = GraphEdge::DOWN; break;
        case GraphEdge::LEFT:        result = GraphEdge::RIGHT; break;
        case GraphEdge::LOWER_LEFT:  result = GraphEdge::UPPER_RIGHT; break;
        case GraphEdge::UPPER_RIGHT: result = GraphEdge::LOWER_LEFT; break;
        case GraphEdge::RIGHT:       result = GraphEdge::LEFT; break;
    }
    return result;
}

GraphEdge RegionConstructor::getDirectionOfNextNeighbor(const PixelGraph::edge_type& edges, GraphEdge direction)
{
    GraphEdge startingDirection = direction;
    do
    {
        rotateDirectionCounterClockwiseByOne(direction);
    } while( direction != startingDirection && !isThereAnEdge(edges,direction) );
    return direction;
}

std::vector<RegionPoint> RegionConstructor::createALonelyPixelPolygonPointPath(const int idx)
{
    std::vector<RegionPoint> polygon;
    polygon.emplace_back(RegionPoint{idx, idx});
    polygon.emplace_back(RegionPoint{idx, idx + widthOfGraph});
    polygon.emplace_back(RegionPoint{idx, idx + widthOfGraph + 1});
    polygon.emplace_back(RegionPoint{idx, idx + 1});
    return polygon;
}

GraphEdge RegionConstructor::get4DirectionWithNoNeighbor(PixelGraph::edge_type edges, GraphEdge initialDirection)
{
    GraphEdge direction = initialDirection;
    do
    {
        rotateDirectionCounterClockwiseByTwo(direction);
    } while( direction != initialDirection && isThereAnEdge(edges,direction) );
    return direction;
}

void RegionConstructor::runRadialSweeps()
{
    for(int idx = 0; idx < wasNodeVisited.size(); ++idx)
    {
        if(!wasNodeVisited[idx])
        {
            std::cout << "\n Looking through " << idx << "\n";
            auto pathForCurrentNode = getPathOfPointsForGivenNode(idx);
            if( !pathForCurrentNode.empty() )
                createdPointPaths.push_back(pathForCurrentNode);
        }
    }

    std::cout << "\n Got " << createdPointPaths.size() << " point paths\n";
}

std::vector<RegionPoint> RegionConstructor::getPathOfPointsForGivenNode(int rootIdx)
{
    std::vector<RegionPoint> result;

    int currentIdx = rootIdx;
    PixelGraph::edge_type currentNodeData = graphData[currentIdx];

    GraphEdge directionToHitEdge = get4DirectionWithNoNeighbor(currentNodeData, GraphEdge::UP);
    while( isThereAnEdge(currentNodeData, directionToHitEdge) )
    {
        currentIdx = getNeighborIdx(currentIdx, directionToHitEdge);
        std::cout << "\n Going through phantom " << currentIdx << "\n";
        currentNodeData = graphData[currentIdx];
        directionToHitEdge = get4DirectionWithNoNeighbor(currentNodeData, directionToHitEdge);
    }
    if(wasNodeVisited[currentIdx])
    {
        wasNodeVisited[rootIdx] = true;
        return result;
    }

    GraphEdge previousDirection = getDirectionOfNextNeighbor(currentNodeData, directionToHitEdge);

    if(previousDirection == directionToHitEdge)
    {
        wasNodeVisited[currentIdx] = true;
        std::cout << "\n" << currentIdx << ", " << currentIdx << " -- lonely polygon\n";
        return createALonelyPixelPolygonPointPath(currentIdx);
    }

    int previousIdx = currentIdx;
    currentIdx = getNeighborIdx(currentIdx, previousDirection);
    currentNodeData = graphData[currentIdx];
    std::cout << "\n Going through phantom " << currentIdx << "\n";

    //TODO change to while(true)
    int k = 0;
    while(k++ < 999999)
    {
        GraphEdge nextDirection = getDirectionOfNextNeighbor(currentNodeData, getInvertedDirection(previousDirection));
        previousIdx = currentIdx;
        wasNodeVisited[previousIdx] = true;
        currentIdx = getNeighborIdx(currentIdx, nextDirection);

        currentNodeData = graphData[currentIdx];
        std::vector<RegionPoint> nextPartOfPath = getNextPathPoints(previousDirection, nextDirection,
                                                                    previousIdx, currentIdx);
        if( result.size() > 0 )
        {
            if(std::find(nextPartOfPath.begin(), nextPartOfPath.end(), result.at(0)) != nextPartOfPath.end())
                break;
        }
        std::cout << "\n" << previousIdx << " -> " << currentIdx << "\n";
        result.insert(result.end(), nextPartOfPath.begin(), nextPartOfPath.end());

        previousDirection = nextDirection;
    }
    return result;
}

int RegionConstructor::getNeighborIdx(const int idx, const GraphEdge direction)
{
    int result;
    switch(direction)
    {
        case GraphEdge::DOWN:        result = idx + widthOfGraph; break;
        case GraphEdge::LOWER_RIGHT: result = idx + widthOfGraph + 1; break;
        case GraphEdge::UPPER_LEFT:  result = idx - widthOfGraph - 1; break;
        case GraphEdge::UP:          result = idx - widthOfGraph; break;
        case GraphEdge::LEFT:        result = idx - 1; break;
        case GraphEdge::LOWER_LEFT:  result = idx + widthOfGraph - 1; break;
        case GraphEdge::UPPER_RIGHT: result = idx - widthOfGraph + 1; break;
        case GraphEdge::RIGHT:       result = idx + 1; break;
    }

    if(result < 0 || result >= wasNodeVisited.size())
    {
        throw std::out_of_range("Neighbor idx out of graph area!");
    }

    return result;
}

void RegionConstructor::addPointsForUpperLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                                    const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{currentIdx+1, currentIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            break;
        case GraphEdge::RIGHT:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                               const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{currentIdx+1, currentIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        default:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForLowerLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                                    const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            break;
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, previousIdx+widthOfGraph});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForDownOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                               const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx-1, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{currentIdx, previousIdx+widthOfGraph});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        default:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForLowerRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                                     const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx+1, previousIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            throw std::invalid_argument("Impossible direction combination for point path!");
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
    }
}

void RegionConstructor::addPointsForRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                                const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx+1, previousIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{currentIdx-1, currentIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{previousIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+widthOfGraph+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        default:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForUpperRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                                     const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx+1, previousIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, currentIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::LOWER_RIGHT:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph});
            break;
        case GraphEdge::RIGHT:
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{previousIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        default:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

void RegionConstructor::addPointsForUpOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction,
                                             const int& previousIdx, const int& currentIdx)
{
    switch(direction)
    {
        case GraphEdge::UPPER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx+1, previousIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+1});
            break;
        case GraphEdge::LOWER_LEFT:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{previousIdx-1, currentIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::DOWN:
            path.emplace_back(RegionPoint{previousIdx, previousIdx+1});
            path.emplace_back(RegionPoint{previousIdx, previousIdx});
            path.emplace_back(RegionPoint{currentIdx, currentIdx});
            break;
        case GraphEdge::UPPER_RIGHT:
            path.emplace_back(RegionPoint{currentIdx+1, previousIdx+1});
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        case GraphEdge::UP:
            path.emplace_back(RegionPoint{currentIdx, currentIdx+widthOfGraph+1});
            break;
        default:
            throw std::invalid_argument("Impossible direction combination for point path!");
    }
}

std::vector<RegionPoint> RegionConstructor::getNextPathPoints(const GraphEdge& previousDirection,
                                                              const GraphEdge& nextDirection, const int& previousIdx,
                                                              const int& currentIdx)
{
    std::vector<RegionPoint> result;

    switch(previousDirection)
    {
        case GraphEdge::UPPER_LEFT:  addPointsForUpperLeftOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::LEFT:        addPointsForLeftOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::LOWER_LEFT:  addPointsForLowerLeftOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::DOWN:        addPointsForDownOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::LOWER_RIGHT: addPointsForLowerRightOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::RIGHT:       addPointsForRightOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::UPPER_RIGHT: addPointsForUpperRightOrigin(result, nextDirection, previousIdx, currentIdx); break;
        case GraphEdge::UP:          addPointsForUpOrigin(result, nextDirection, previousIdx, currentIdx); break;
    }

    return result;
}

const std::vector<std::vector<RegionPoint> >& RegionConstructor::getCreatedPointPaths() const
{
    return createdPointPaths;
}
