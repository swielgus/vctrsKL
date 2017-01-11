//
// Created by sw on 11.01.17.
//

#ifndef VCTRSKL_IMAGEDATA_HPP
#define VCTRSKL_IMAGEDATA_HPP

#include <cstddef>
#include <png++/png.hpp>

class ImageData
{
public:
    std::size_t getWidth() const;
    std::size_t getHeight() const;
private:
    png::image<png::rgb_pixel> internalImage;
};

#endif //VCTRSKL_IMAGEDATA_HPP
