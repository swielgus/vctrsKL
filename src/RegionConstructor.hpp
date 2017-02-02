#ifndef VCTRSKL_REGIONCONSTRUCTOR_HPP
#define VCTRSKL_REGIONCONSTRUCTOR_HPP

#include <vector>
#include <clipper.hpp>
#include <deque>
#include "PixelGraph.hpp"
#include "PolygonSide.hpp"
#include "Constants.hpp"

class RegionConstructor
{
public:
    RegionConstructor() = delete;
    RegionConstructor(const std::vector<PolygonSide>& usedSideData,
                      const std::vector<PixelGraph::edge_type>&& usedGraphData,
                      unsigned int usedWidth, unsigned int usedHeight);
    ~RegionConstructor();

    const ClipperLib::Paths& getBoundaries() const;
    std::vector< std::vector<PathPoint> > createPathPoints();
private:
    const std::vector<PolygonSide>&          polygonSideData;
    const std::vector<PixelGraph::edge_type> graphData;
    unsigned int                             width;
    unsigned int                             height;
    std::vector<bool>                        wasNodeVisited;
    ClipperLib::Paths                        generatedBoundaries;

    void generateBoundariesByRadialSweeps();
    void markComponentAsVisited(const int rootRow, const int rootCol);
    void markNeighBorAsVisitedAndAddToQueue(const int& currentIdx, const GraphEdge direction,
                                            std::deque<int>& queueOfNodesToVisit);
    bool isThereAnEdge(const PixelGraph::edge_type& currentNodeData, const GraphEdge& direction) const;
    int getNeighborIdx(const int& idx, const GraphEdge direction);
    int getNeighborRowIdx(int row, GraphEdge direction);
    int getNeighborColIdx(int col, GraphEdge direction);
    GraphEdge get4DirectionWithNoNeighbor(const PixelGraph::edge_type& edges, GraphEdge initialDirection);
    GraphEdge getDirectionOfNextNeighbor(const PixelGraph::edge_type& edges, GraphEdge direction);
    void rotateDirectionCounterClockwiseByOne(GraphEdge& direction);
    void rotateDirectionCounterClockwiseByTwo(GraphEdge& direction);
    void addUpperLeftCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon);
    void addLowerLeftCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon);
    void addUpperRightCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon);
    void addLowerRightCornerOfPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Path& polygon);
    void addPolygonToPath(const unsigned int& row, const unsigned int& col, ClipperLib::Paths& polygonChain);
    GraphEdge getInvertedDirection(const GraphEdge& direction) const;
    ClipperLib::Paths createChainOfPolygonsForRoot(const unsigned int& row, const unsigned int& col);
};


#endif //VCTRSKL_REGIONCONSTRUCTOR_HPP
