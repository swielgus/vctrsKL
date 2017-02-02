#include <algorithm>
#include <deque>
#include <iostream>
#include "RegionConstructor.hpp"

RegionConstructor::RegionConstructor(const std::vector<PolygonSide>& usedSideData,
                                     const std::vector<PixelGraph::edge_type>&& usedGraphData,
                                     unsigned int usedWidth, unsigned int usedHeight)
        : polygonSideData{usedSideData}, graphData{std::move(usedGraphData)}, width{usedWidth}, height{usedHeight},
          wasNodeVisited{}, generatedBoundaries{}
{
    wasNodeVisited.resize(width * height, false);
    generateBoundariesByRadialSweeps();
}

RegionConstructor::~RegionConstructor()
{
}

void RegionConstructor::generateBoundariesByRadialSweeps()
{
    std::vector<ClipperLib::Paths> gatheredPolygonChains;
    for(unsigned int row = 0; row < height; ++row)
    for(unsigned int col = 0; col < width; ++col)
    {
        unsigned int idx = col + row * width;
        if(!wasNodeVisited[idx])
        {
            //std::cout << "\n" << row << "," << col << " = ";
            ClipperLib::Paths currentPolygonChain = createChainOfPolygonsForRoot(row, col);
            if( !currentPolygonChain.empty() )
                gatheredPolygonChains.push_back(std::move(currentPolygonChain));
            //std::cout << "\n";
        }
    }

    for(const auto& chain : gatheredPolygonChains)
    {
        ClipperLib::PolyTree unionResult;
        ClipperLib::Clipper clipTool;
        clipTool.PreserveCollinear(true);
        clipTool.AddPaths(chain, ClipperLib::PolyType::ptSubject, true);
        clipTool.Execute(ClipperLib::ClipType::ctUnion, unionResult, ClipperLib::PolyFillType::pftPositive,
                         ClipperLib::PolyFillType::pftPositive);

        if( unionResult.GetFirst() )
        {
            /*std::cout << "\n";
            for(const auto& point : unionResult.GetFirst()->Contour)
                std::cout << point.X << "," << point.Y << "|";*/
            generatedBoundaries.push_back(std::move(unionResult.GetFirst()->Contour));
        }
    }
}

ClipperLib::Paths RegionConstructor::createChainOfPolygonsForRoot(const unsigned int& row, const unsigned int& col)
{
    ClipperLib::Paths result;

    int currentRow = row;
    int currentCol = col;
    int currentIdx = col + row * width;
    PixelGraph::edge_type currentNodeData = graphData[currentIdx];

    GraphEdge previousDirection = GraphEdge::UP;
    GraphEdge directionToHitEdge = get4DirectionWithNoNeighbor(currentNodeData, previousDirection);
    while( isThereAnEdge(currentNodeData, directionToHitEdge) )
    {
        currentRow = getNeighborRowIdx(currentRow, directionToHitEdge);
        currentCol = getNeighborColIdx(currentCol, directionToHitEdge);
        currentIdx = col + row * width;
        currentNodeData = graphData[currentIdx];
        if(wasNodeVisited[currentIdx])
        {
            markComponentAsVisited(row, col);
            return result;
        }
        directionToHitEdge = get4DirectionWithNoNeighbor(currentNodeData, directionToHitEdge);
    }

    previousDirection = getDirectionOfNextNeighbor(currentNodeData, directionToHitEdge);
    if(previousDirection == directionToHitEdge)
    {
        wasNodeVisited[currentIdx] = true;
        //std::cout << "\n" << currentIdx << ", " << currentIdx << " -- lonely polygon\n";
        addPolygonToPath(row, col, result);
        return result;
    }

    int previousRow = currentRow;
    int previousCol = currentCol;
    int previousIdx = currentIdx;
    currentRow = getNeighborRowIdx(previousRow, previousDirection);
    currentCol = getNeighborColIdx(previousCol, previousDirection);
    currentIdx = currentCol + currentRow * width;
    currentNodeData = graphData[currentIdx];

    int idxBeforeFirst = currentIdx;
    std::vector<int> sequenceOfPoints;

    while(true)
    {
        GraphEdge nextDirection = getDirectionOfNextNeighbor(currentNodeData, getInvertedDirection(previousDirection));
        previousRow = currentRow;
        previousCol = currentCol;
        previousIdx = currentIdx;

        currentRow = getNeighborRowIdx(previousRow, nextDirection);
        currentCol = getNeighborColIdx(previousCol, nextDirection);
        currentIdx = currentCol + currentRow * width;
        currentNodeData = graphData[currentIdx];

        auto positionWhereCurrentIdxIsRepeated = std::find(sequenceOfPoints.begin(), sequenceOfPoints.end(), currentIdx);
        if(positionWhereCurrentIdxIsRepeated != sequenceOfPoints.end())
        {
            bool didCurrentIdxOccurBefore = (previousIdx == idxBeforeFirst);
            if(positionWhereCurrentIdxIsRepeated != sequenceOfPoints.begin())
                didCurrentIdxOccurBefore = (previousIdx == *(--positionWhereCurrentIdxIsRepeated));

            if(didCurrentIdxOccurBefore)
                break;
        }
        sequenceOfPoints.push_back(currentIdx);

        addPolygonToPath(currentRow, currentCol, result);
        previousDirection = nextDirection;
    }

    markComponentAsVisited(row, col);

    return result;
}

void
RegionConstructor::addPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Paths& polygonChain)
{
    //std::cout << "[" << row << ","<< col << "] - ";
    ClipperLib::Path polygon;

    addUpperLeftCornerOfPolygonToPath(row, col, polygon);
    addLowerLeftCornerOfPolygonToPath(row, col, polygon);
    addLowerRightCornerOfPolygonToPath(row, col, polygon);
    addUpperRightCornerOfPolygonToPath(row, col, polygon);

    /*std::cout << "-|";
    for(const auto& point : polygon)
        std::cout << point.X << "," << point.Y << "|";
*/
    polygonChain.push_back(std::move(polygon));
}

void RegionConstructor::addUpperLeftCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon)
{
    const int idx = col + row * width;
    const auto& nodeDetails = polygonSideData[idx];
    if( nodeDetails.getType() == PolygonSide::Type::Backslash )
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
    }
    else if( nodeDetails.getType() == PolygonSide::Type::ForwardSlash )
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
    }
    else
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
    }
}

void RegionConstructor::addLowerLeftCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon)
{
    if(col == 0 || row == height-1)
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(100*(row+1)), static_cast<int>(100*col));
    }
    else
    {
        const int idx = col + (row+1) * width;
        const auto& nodeDetails = polygonSideData[idx];
        if( nodeDetails.getType() == PolygonSide::Type::Backslash )
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
        }
        else
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        }
    }
}

void RegionConstructor::addUpperRightCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon)
{
    if(col == width-1 || row == 0)
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(100*row), static_cast<int>(100*(col+1)));
    }
    else
    {
        const int idx = col+1 + row * width;
        const auto& nodeDetails = polygonSideData[idx];
        if( nodeDetails.getType() == PolygonSide::Type::Backslash )
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        }
        else if( nodeDetails.getType() == PolygonSide::Type::ForwardSlash )
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
        }
        else
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        }
    }
}

void RegionConstructor::addLowerRightCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon)
{
    if(col == width-1 || row == height-1)
    {
        polygon << ClipperLib::IntPoint(static_cast<int>(100*(row+1)), static_cast<int>(100*(col+1)));
    }
    else
    {
        const int idx = col+1 + (row+1) * width;
        const auto& nodeDetails = polygonSideData[idx];
        if( nodeDetails.getType() == PolygonSide::Type::ForwardSlash )
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointB[0]), static_cast<int>(nodeDetails.pointB[1]));
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        }
        else
        {
            polygon << ClipperLib::IntPoint(static_cast<int>(nodeDetails.pointA[0]), static_cast<int>(nodeDetails.pointA[1]));
        }
    }
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

GraphEdge RegionConstructor::get4DirectionWithNoNeighbor(const PixelGraph::edge_type& edges, GraphEdge initialDirection)
{
    GraphEdge direction = initialDirection;
    do
    {
        rotateDirectionCounterClockwiseByTwo(direction);
    } while( direction != initialDirection && isThereAnEdge(edges,direction) );
    return direction;
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

void RegionConstructor::markComponentAsVisited(const int rootRow, const int rootCol)
{
    int currentIdx = rootCol + rootRow * width;
    wasNodeVisited[currentIdx] = true;
    std::deque<int> queueOfNodesToVisit;
    queueOfNodesToVisit.push_back(currentIdx);
    while( !queueOfNodesToVisit.empty() )
    {
        currentIdx = queueOfNodesToVisit.front();
        queueOfNodesToVisit.pop_front();

        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::UPPER_LEFT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::LEFT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::LOWER_LEFT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::DOWN, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::LOWER_RIGHT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::RIGHT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::UPPER_RIGHT, queueOfNodesToVisit);
        markNeighBorAsVisitedAndAddToQueue(currentIdx, GraphEdge::UP, queueOfNodesToVisit);
    }
}

void RegionConstructor::markNeighBorAsVisitedAndAddToQueue(const int& currentIdx, const GraphEdge direction,
                                                           std::deque<int>& queueOfNodesToVisit)
{
    PixelGraph::edge_type currentNodeData = graphData[currentIdx];
    if( isThereAnEdge(currentNodeData, direction) )
    {
        int neighborIdx = getNeighborIdx(currentIdx, direction);
        if(!wasNodeVisited[neighborIdx])
        {
            wasNodeVisited[neighborIdx] = true;
            queueOfNodesToVisit.push_back(neighborIdx);
        }
    }
}

bool RegionConstructor::isThereAnEdge(const PixelGraph::edge_type& currentNodeData, const GraphEdge& direction) const
{
    return currentNodeData & static_cast<PixelGraph::edge_type>(direction);
}

int RegionConstructor::getNeighborRowIdx(int row, GraphEdge direction)
{
    if(direction == GraphEdge::UP || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::UPPER_RIGHT)
        row--;
    if(direction == GraphEdge::DOWN || direction == GraphEdge::LOWER_LEFT || direction == GraphEdge::LOWER_RIGHT)
        row++;

    if(row < 0 || row >= height)
    {
        throw std::out_of_range("Neighbor row out of graph area!");
    }

    return row;
}

int RegionConstructor::getNeighborColIdx(int col, GraphEdge direction)
{
    if(direction == GraphEdge::LEFT || direction == GraphEdge::UPPER_LEFT || direction == GraphEdge::LOWER_LEFT)
        col--;
    if(direction == GraphEdge::RIGHT || direction == GraphEdge::UPPER_RIGHT || direction == GraphEdge::LOWER_RIGHT)
        col++;

    if(col < 0 || col >= width)
    {
        throw std::out_of_range("Neighbor col out of graph area!");
    }

    return col;
}

int RegionConstructor::getNeighborIdx(const int& idx, const GraphEdge direction)
{
    int result;
    switch(direction)
    {
        case GraphEdge::DOWN:        result = idx + width; break;
        case GraphEdge::LOWER_RIGHT: result = idx + width + 1; break;
        case GraphEdge::UPPER_LEFT:  result = idx - width - 1; break;
        case GraphEdge::UP:          result = idx - width; break;
        case GraphEdge::LEFT:        result = idx - 1; break;
        case GraphEdge::LOWER_LEFT:  result = idx + width - 1; break;
        case GraphEdge::UPPER_RIGHT: result = idx - width + 1; break;
        case GraphEdge::RIGHT:       result = idx + 1; break;
    }

    if(result < 0 || result >= wasNodeVisited.size())
    {
        throw std::out_of_range("Neighbor idx out of graph area!");
    }

    return result;
}

const ClipperLib::Paths& RegionConstructor::getBoundaries() const
{
    return generatedBoundaries;
}

std::vector<std::vector<PathPoint> > RegionConstructor::createPathPoints()
{
    std::vector< std::vector<PathPoint> > result;
    result.resize(generatedBoundaries.size());

    for(int i = 0; i < generatedBoundaries.size(); ++i)
    {
        const auto& currentPathSource = generatedBoundaries.at(i);
        auto& currentPath = result.at(i);
        for(const auto& pathElement : currentPathSource)
        {
            PathPoint currentPoint{false, pathElement.X / 100, pathElement.Y / 100};
            unsigned int rowRemainder = pathElement.X % 100;
            unsigned int colRemainder = pathElement.Y % 100;
            unsigned int rowQuotient = (pathElement.X - rowRemainder) / 100;
            unsigned int colQuotient = (pathElement.Y - colRemainder) / 100;

            if(rowRemainder == 25)
            {
                currentPoint.useBPoint = true;
                currentPoint.rowOfCoordinates = rowQuotient;
            }
            else if(rowRemainder == 75)
            {
                currentPoint.rowOfCoordinates = rowQuotient + 1;
            }

            if(colRemainder == 25)
            {
                currentPoint.colOfCoordinates = colQuotient;
            }
            else if(colRemainder == 75)
            {
                currentPoint.colOfCoordinates = colQuotient + 1;
            }

            currentPath.push_back(std::move(currentPoint));
        }
    }

    return result;
}
