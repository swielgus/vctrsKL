#ifndef VCTRSKL_REGIONCONSTRUCTOR_HPP
#define VCTRSKL_REGIONCONSTRUCTOR_HPP

#include "PixelGraph.hpp"
#include "Constants.hpp"

class RegionConstructor
{
    bool isThereAnEdge(const PixelGraph::edge_type& currentNodeData, const GraphEdge& direction) const;
    void rotateDirectionCounterClockwiseByOne(GraphEdge& direction);
    void rotateDirectionCounterClockwiseByTwo(GraphEdge& direction);
    GraphEdge getDirectionOfNextNeighbor(const PixelGraph::edge_type& edges, GraphEdge direction);
    GraphEdge getInvertedDirection(const GraphEdge& direction) const;
    std::vector<RegionPoint> createALonelyPixelPolygonPointPath(const int idxOfCorrespondingNode);

    GraphEdge get4DirectionWithNoNeighbor(PixelGraph::edge_type edges, GraphEdge initialDirection);
    void addPointsForUpperLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                     const int& currentIdx);
    void addPointsForLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                const int& currentIdx);
    void addPointsForLowerLeftOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                     const int& currentIdx);
    void addPointsForDownOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                const int& currentIdx);
    void addPointsForLowerRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                      const int& currentIdx);
    void addPointsForRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                 const int& currentIdx);
    void addPointsForUpperRightOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                                      const int& currentIdx);
    void addPointsForUpOrigin(std::vector<RegionPoint>& path, const GraphEdge& direction, const int& previousIdx,
                              const int& currentIdx);
    int getNeighborIdx(const int idx, const GraphEdge direction);
    std::vector<RegionPoint> getNextPathPoints(const GraphEdge& edge, const GraphEdge& graphEdge, const int& idx,
                                               const int& currentIdx);
    void runRadialSweeps();
    std::vector<RegionPoint> getPathOfPointsForGivenNode(int rootIdx);

    const std::vector<PixelGraph::edge_type> graphData;
    int widthOfGraph;
    int heightOfGraph;
    std::vector< std::vector<RegionPoint> > createdPointPaths;
    std::vector<bool> wasNodeVisited;
public:
    RegionConstructor();
    RegionConstructor(const std::vector<PixelGraph::edge_type>&& usedGraphData, int width, int height);
    ~RegionConstructor();

    const std::vector< std::vector<RegionPoint> >& getCreatedPointPaths() const;
};


#endif //VCTRSKL_REGIONCONSTRUCTOR_HPP
