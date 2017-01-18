#ifndef VCTRSKL_PIXELGRAPH_HPP
#define VCTRSKL_PIXELGRAPH_HPP

#include "ImageData.hpp"

class PixelGraph
{
public:
    using color_type = Color::color_byte;

    PixelGraph(const ImageData& image);
    ~PixelGraph();

    void resolveUnnecessaryDiagonals();
    std::vector<std::vector<color_type> > getEdgeValues() const;
private:
    const ImageData& sourceImage;
    color_type* d_pixelConnections;
    color_type* d_pixelDirections;

    void freeDeviceData();
    void constructGraph();
};


#endif //VCTRSKL_PIXELGRAPH_HPP
