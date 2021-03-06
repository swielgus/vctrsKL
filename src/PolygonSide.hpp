#ifndef VCTRSKL_POLYGONSIDE_HPP
#define VCTRSKL_POLYGONSIDE_HPP

#include "Constants.hpp"

struct PolygonSide
{
    using point_type = Polygon::cord_type;
    using info_type = Polygon::byte;
    enum class Type : info_type
    {
        Point        = 0,
        Backslash    = 1,
        ForwardSlash = 2
    };

    info_type info;
    point_type pointA[2];
    point_type pointB[2];

    Type getType() const;
    int getNumberOfRegionsUsingA() const;
    int getNumberOfRegionsUsingB() const;
    void increaseNumberOfRegionsUsingAByOne();
    void increaseNumberOfRegionsUsingBByOne();

    /*PolygonSide(const Type& type, const point_type& aRow, const point_type& aCol,
                const point_type& bRow = 0.0, const point_type& bCol = 0.0);*/
};

struct PathPoint
{
    bool useBPoint;
    int  rowOfCoordinates;
    int  colOfCoordinates;

    friend inline bool operator== (const PathPoint& a, const PathPoint& b)
    {
        return a.useBPoint == b.useBPoint &&
               a.rowOfCoordinates == b.rowOfCoordinates &&
               a.colOfCoordinates == b.colOfCoordinates;
    }
    friend inline bool operator!= (const PathPoint& a, const PathPoint& b)
    {
        return !(a == b);
    }
};

#endif //VCTRSKL_POLYGONSIDE_HPP
