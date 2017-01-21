#include "PixelGraphInfo.hpp"

PixelGraphInfo::PixelGraphInfo(edge_type* const& pixelConnections, const std::size_t& imgWidth,
                               const std::size_t& imgHeight)
    : edges{pixelConnections}, width{imgWidth}, height{imgHeight}
{}