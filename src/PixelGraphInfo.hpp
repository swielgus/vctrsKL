#ifndef VCTRSKL_PIXELGRAPHINFO_HPP
#define VCTRSKL_PIXELGRAPHINFO_HPP

#include "Constants.hpp"

struct PixelGraphInfo
{
    using edge_type = Graph::byte;
    using color_type = Color::byte;

    PixelGraphInfo() = delete;
    PixelGraphInfo(edge_type* const& pixelConnections, const std::size_t& imgWidth,
                   const std::size_t& imgHeight);

    edge_type* edges;
    std::size_t width;
    std::size_t height;
};


#endif //VCTRSKL_PIXELGRAPHINFO_HPP
