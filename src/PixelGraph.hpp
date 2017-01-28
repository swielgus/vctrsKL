#ifndef VCTRSKL_PIXELGRAPH_HPP
#define VCTRSKL_PIXELGRAPH_HPP

#include "ImageData.hpp"

class PixelGraph
{
public:
    using color_type = Color::byte;
    using edge_type = Graph::byte;

    PixelGraph() = delete;
    PixelGraph(const ImageData& image);
    ~PixelGraph();

    void resolveUnnecessaryDiagonals();
    void resolveDisconnectingDiagonals();
    void resolveCrossings();
    std::vector<std::vector<edge_type> > getEdgeValues() const;
    std::vector<edge_type> get1DEdgeValues() const;
    const edge_type* getGPUAddressOfGraphData() const;
    std::size_t getWidth() const;
    std::size_t getHeight() const;
private:
    const ImageData& sourceImage;
    edge_type* d_pixelConnections;

    void freeDeviceData();
    void constructGraph();
};


#endif //VCTRSKL_PIXELGRAPH_HPP
