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

namespace Cell
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

enum class CellSide : Cell::byte
{
    /*
     * TYPE A: /
     * TYPE B: \
     * TYPE C: .
     */
    UPPER_LEFT_TYPE_A = 1 << 0,
    UPPER_LEFT_TYPE_B = 2 << 0,
    UPPER_LEFT_TYPE_C = 3 << 0,

    LOWER_LEFT_TYPE_A = 1 << 2,
    LOWER_LEFT_TYPE_B = 2 << 2,
    LOWER_LEFT_TYPE_C = 3 << 2,

    LOWER_RIGHT_TYPE_A = 1 << 4,
    LOWER_RIGHT_TYPE_B = 2 << 4,
    LOWER_RIGHT_TYPE_C = 3 << 4,

    UPPER_RIGHT_TYPE_A = 1 << 6,
    UPPER_RIGHT_TYPE_B = 2 << 6,
    UPPER_RIGHT_TYPE_C = 3 << 6
};

#endif //VCTRSKL_CONSTANTS_HPP
