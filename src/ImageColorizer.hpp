#ifndef VCTRSKL_IMAGECOLORIZER_HPP
#define VCTRSKL_IMAGECOLORIZER_HPP

#include "PolygonSideMap.hpp"

class ImageColorizer
{
public:
    ImageColorizer() = delete;
    ImageColorizer(const PolygonSideMap& usedSideMap);

    const std::vector< std::vector<std::size_t> >& getPointRegionIdxValues() const;
private:
    const ClipperLib::Paths& regionBoundaries;
    const unsigned int imageWidth;
    const unsigned int imageHeight;
    std::vector< std::vector<std::size_t> > regionIdxOfPoints;
};


#endif //VCTRSKL_IMAGECOLORIZER_HPP
