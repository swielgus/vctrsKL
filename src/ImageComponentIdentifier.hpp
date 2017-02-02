#ifndef VCTRSKL_IMAGECOMPONENTIDENTIFIER_HPP
#define VCTRSKL_IMAGECOMPONENTIDENTIFIER_HPP

#include <vector>
#include <deque>
#include "Constants.hpp"

class ImageComponentIdentifier
{
public:
    using color_type = Color::byte;
    ~ImageComponentIdentifier();
    ImageComponentIdentifier() = delete;
    ImageComponentIdentifier(const std::vector<color_type>& colors, unsigned int width, unsigned int height);
    const std::vector<int>& getComponentLabels();
private:
    const std::vector<color_type>& colorsYUV;
    std::vector<bool> wasThisPointVisited;
    std::vector<int> componentLabels;
    unsigned int widthOfImage;
    unsigned int heightOfImage;

    void markNeighborWithTheLabelAndAddToQueue(int row, int col, int neighboredIdx, const int rootLabel,
                                               std::deque<int>& rowsToVisit, std::deque<int>& colsToVisit);
    bool areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV, const Color::byte& bY,
                             const Color::byte& bU, const Color::byte& bV);

    void setComponentLabelsForRoot(int row, int col);
};


#endif //VCTRSKL_IMAGECOMPONENTIDENTIFIER_HPP
