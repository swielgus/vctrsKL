#ifndef VCTRSKL_POLYGONSIDE_HPP
#define VCTRSKL_POLYGONSIDE_HPP

#include "Constants.hpp"

struct PolygonSide
{
    using point_type = Polygon::cord_type;
    using info_type = Polygon::byte;
    enum class Type : info_type
    {
        Point        = 1,
        Backslash    = 2,
        ForwardSlash = 3
    };

    info_type info;
    point_type pointA[2];
    point_type pointB[2];

    Type getType() const;

    /*PolygonSide(const Type& type, const point_type& aRow, const point_type& aCol,
                const point_type& bRow = 0.0, const point_type& bCol = 0.0);*/
};


#endif //VCTRSKL_POLYGONSIDE_HPP
