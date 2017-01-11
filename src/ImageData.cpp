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