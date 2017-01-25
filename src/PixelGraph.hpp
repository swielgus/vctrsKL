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
    std::vector<std::vector<color_type> > getEdgeValues() const;
private:
    const ImageData& sourceImage;
    edge_type* d_pixelConnections;

    void freeDeviceData();
    void constructGraph();
};


#endif //VCTRSKL_PIXELGRAPH_HPP
