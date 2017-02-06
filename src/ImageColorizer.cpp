#include "ImageColorizer.hpp"

ImageColorizer::ImageColorizer(const PolygonSideMap& usedSideMap)
    : regionBoundaries{usedSideMap.getGeneratedRegionBoundaries()}, imageWidth{usedSideMap.getImageWidth()},
      imageHeight{usedSideMap.getImageHeight()}, regionIdxOfPoints{}
{
    regionIdxOfPoints.resize(imageHeight);
    for(auto& row : regionIdxOfPoints)
        row.resize(imageWidth, -1);

    for(int row = imageHeight-1; row >= 0; --row)
    for(int col = imageWidth-1; col >= 0; --col)
    {
        ClipperLib::IntPoint pointCenter{100 * row + 50, 100 * col + 50};

        int idxOfPath = regionBoundaries.size();
        bool wasAPathFound = false;
        while(!wasAPathFound && idxOfPath >= 0)
        {
            --idxOfPath;
            wasAPathFound = (ClipperLib::PointInPolygon(pointCenter, regionBoundaries[idxOfPath]) != 0);
        }

        if(wasAPathFound)
            regionIdxOfPoints[row][col] = idxOfPath;
        else
            throw std::invalid_argument("No path found for a point!");
    }
}

const std::vector<std::vector<std::size_t> >& ImageColorizer::getPointRegionIdxValues() const
{
    return regionIdxOfPoints;
}
