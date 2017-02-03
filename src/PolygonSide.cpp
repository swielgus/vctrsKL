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
    const int FIRST_TWO_BITS = 3;
    return static_cast<PolygonSide::Type>( info & FIRST_TWO_BITS );
}

int PolygonSide::getNumberOfRegionsUsingA() const
{
    const int LAST_THREE_BITS = 224;
    return ((info & LAST_THREE_BITS) >> 5);
}

int PolygonSide::getNumberOfRegionsUsingB() const
{
    const int SECOND_THREE_BITS = 28;
    return ((info & SECOND_THREE_BITS) >> 2);
}

void PolygonSide::increaseNumberOfRegionsUsingAByOne()
{
    const int ALL_BUT_LAST_THREE_BITS = 31;
    int result = getNumberOfRegionsUsingA() + 1;
    info &= ALL_BUT_LAST_THREE_BITS;
    info |= (result << 5);
}

void PolygonSide::increaseNumberOfRegionsUsingBByOne()
{
    const int ALL_BUT_SECOND_THREE_BITS = 227;
    int result = getNumberOfRegionsUsingB() + 1;
    info &= ALL_BUT_SECOND_THREE_BITS;
    info |= (result << 2);
}
