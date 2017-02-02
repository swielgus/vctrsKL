#include <cstdlib>
#include <iostream>
#include "ImageComponentIdentifier.hpp"

ImageComponentIdentifier::ImageComponentIdentifier(const std::vector<ImageComponentIdentifier::color_type>& colors,
                                                   unsigned int width, unsigned int height)
    : colorsYUV{colors}, wasThisPointVisited{}, componentLabels{}, widthOfImage{width}, heightOfImage{height}
{
    wasThisPointVisited.resize(widthOfImage * heightOfImage, false);
    componentLabels.resize(widthOfImage * heightOfImage);

    for(int row = 0; row < heightOfImage; ++row)
    for(int col = 0; col < widthOfImage; ++col)
    {
        //std::cout << "\n" << row << "," << col << "\n";
        setComponentLabelsForRoot(row, col);
    }
}

void ImageComponentIdentifier::markNeighborWithTheLabelAndAddToQueue(int row, int col, int neighboredIdx,
                                                                     const int rootLabel,
                                                                     std::deque<int>& rowsToVisit,
                                                                     std::deque<int>& colsToVisit)
{
    int neighborIdx = col + row * widthOfImage;
    if(!wasThisPointVisited[neighborIdx])
    {
        const color_type& thisPointY = colorsYUV[3 * neighboredIdx + 0];
        const color_type& thisPointU = colorsYUV[3 * neighboredIdx + 1];
        const color_type& thisPointV = colorsYUV[3 * neighboredIdx + 2];
        const color_type& comparedPointY = colorsYUV[3 * neighborIdx + 0];
        const color_type& comparedPointU = colorsYUV[3 * neighborIdx + 1];
        const color_type& comparedPointV = colorsYUV[3 * neighborIdx + 2];
        if(areYUVColorsSimilar(thisPointY, thisPointU, thisPointV, comparedPointY, comparedPointU, comparedPointV))
        {
            wasThisPointVisited[neighborIdx] = true;
            componentLabels[neighborIdx] = rootLabel;
            rowsToVisit.push_back(row);
            colsToVisit.push_back(col);
        }
    }
}

void ImageComponentIdentifier::setComponentLabelsForRoot(int row, int col)
{
    int rootIdx = col + row * widthOfImage;
    if( !wasThisPointVisited[rootIdx] )
    {
        wasThisPointVisited[rootIdx] = true;
        componentLabels[rootIdx] = rootIdx;

        std::deque<int> queueOfRowsToVisit;
        std::deque<int> queueOfColsToVisit;
        queueOfRowsToVisit.push_back(row);
        queueOfColsToVisit.push_back(col);
        while( !queueOfRowsToVisit.empty() && !queueOfColsToVisit.empty() )
        {
            int currentRow = queueOfRowsToVisit.front();
            int currentCol = queueOfColsToVisit.front();
            int currentIdx = currentCol + currentRow * widthOfImage;
            queueOfRowsToVisit.pop_front();
            queueOfColsToVisit.pop_front();

            for(int modRow = -1; modRow <= 1; ++modRow)
            for(int modCol = -1; modCol <= 1; ++modCol)
            {
                int neighborRow = currentRow + modRow;
                int neighborCol = currentCol + modCol;
                if( (neighborRow != 0 || neighborCol != 0) &&
                     neighborRow >= 0 && neighborRow < heightOfImage &&
                     neighborCol >= 0 && neighborCol < widthOfImage)
                {
                    markNeighborWithTheLabelAndAddToQueue(neighborRow, neighborCol, currentIdx, rootIdx,
                                                          queueOfRowsToVisit, queueOfColsToVisit);
                }
            }
        }
    }
}

bool ImageComponentIdentifier::areYUVColorsSimilar(const Color::byte& aY, const Color::byte& aU, const Color::byte& aV,
                                                   const Color::byte& bY, const Color::byte& bU, const Color::byte& bV)
{
    const Color::byte thresholdY = 48;
    const Color::byte thresholdU = 7;
    const Color::byte thresholdV = 6;

    return abs(aY - bY) <= thresholdY && abs(aU - bU) <= thresholdU && abs(aV - bV) <= thresholdV;
}

const std::vector<int>& ImageComponentIdentifier::getComponentLabels()
{
    return componentLabels;
}

ImageComponentIdentifier::~ImageComponentIdentifier()
{

}
