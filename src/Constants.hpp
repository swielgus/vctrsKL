#ifndef VCTRSKL_CONSTANTS_HPP
#define VCTRSKL_CONSTANTS_HPP

#include <cstdint>

namespace Color
{
    using byte = uint8_t;
}

namespace Graph
{
    using byte = uint8_t;
}

enum class GraphEdge : Graph::byte
{
    UPPER_LEFT  = 1 << 0,
    LEFT        = 1 << 1,
    LOWER_LEFT  = 1 << 2,
    DOWN        = 1 << 3,
    LOWER_RIGHT = 1 << 4,
    RIGHT       = 1 << 5,
    UPPER_RIGHT = 1 << 6,
    UP          = 1 << 7
};

#endif //VCTRSKL_CONSTANTS_HPP
