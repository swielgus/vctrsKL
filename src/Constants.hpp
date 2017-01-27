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
    using cord_type = float;
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

enum class CellSideType : Cell::byte
{
    Backslash    = 1,
    ForwardSlash = 2,
    Point        = 3
};

struct CellSide
{
    Cell::byte      type;
    Cell::cord_type pointA[2];
    Cell::cord_type pointB[2];
};

struct RegionPoint
{
    bool usePointBWhenForwardSlash;
    int idxOfCoordinates;
};

#endif //VCTRSKL_CONSTANTS_HPP
