#include "PolygonSide.hpp"

/*PolygonSide::PolygonSide(const PolygonSide::Type& type, const PolygonSide::point_type& aRow,
                         const PolygonSide::point_type& aCol, const PolygonSide::point_type& bRow,
                         const PolygonSide::point_type& bCol)
    : info{static_cast<info_type>(type)}
{
    pointA[0] = aRow;
    pointA[1] = aCol;
    pointB[0] = bRow;
    pointB[1] = bCol;
}*/
PolygonSide::Type PolygonSide::getType() const
{
    const int FIRST_THREE_BITS = 7;
    return static_cast<PolygonSide::Type>( info & FIRST_THREE_BITS );
}
