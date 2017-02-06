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

namespace Polygon
{
    using byte = uint8_t;
    using cord_type = float;
}

namespace Curve
{
    using byte = uint8_t;
    using param_type = float;
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

enum class CornerPatternEdge : Curve::byte
{
    NOTHING,
    LONG_VERTICAL_UP, LONG_VERTICAL_DOWN,                                   // ||
    LONG_HORIZONTAL_LEFT, LONG_HORIZONTAL_RIGHT,                            // __
    SHORT_VERTICAL_UP, SHORT_VERTICAL_DOWN,                                 // |
    SHORT_HORIZONTAL_LEFT, SHORT_HORIZONTAL_RIGHT,                          // -
    VERTICAL_RIGHT_LONG_DIAGONAL_UP, VERTICAL_RIGHT_LONG_DIAGONAL_DOWN,     // /
    VERTICAL_LEFT_LONG_DIAGONAL_UP, VERTICAL_LEFT_LONG_DIAGONAL_DOWN,       // :
    HORIZONTAL_RIGHT_LONG_DIAGONAL_UP, HORIZONTAL_RIGHT_LONG_DIAGONAL_DOWN, // -/
    HORIZONTAL_LEFT_LONG_DIAGONAL_UP, HORIZONTAL_LEFT_LONG_DIAGONAL_DOWN,   // :-
    RIGHT_SHORT_DIAGONAL_UP, RIGHT_SHORT_DIAGONAL_DOWN,                     // / uniform
    LEFT_SHORT_DIAGONAL_UP, LEFT_SHORT_DIAGONAL_DOWN                        // : uniform
};

#endif //VCTRSKL_CONSTANTS_HPP
