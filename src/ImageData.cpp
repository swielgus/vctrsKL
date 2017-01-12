//
// Created by sw on 11.01.17.
//

#include "ImageData.hpp"

std::size_t ImageData::getWidth() const
{
    return internalImage.get_width();
}

std::size_t ImageData::getHeight() const
{
    return internalImage.get_height();
}

void ImageData::loadImage(std::string filename)
{
    internalImage.read(filename);
}

const png::rgb_pixel& ImageData::getPixel(std::size_t x, std::size_t y) const
{
    return internalImage.get_row(x).at(y);
}

const ImageData::color_type& ImageData::getPixelRed(std::size_t x, std::size_t y) const
{
    return getPixel(x,y).red;
}
const ImageData::color_type& ImageData::getPixelGreen(std::size_t x, std::size_t y) const
{
    return getPixel(x,y).green;
}
const ImageData::color_type& ImageData::getPixelBlue(std::size_t x, std::size_t y) const
{
    return getPixel(x,y).blue;
}
