#ifndef VCTRSKL_IMAGEDATA_HPP
#define VCTRSKL_IMAGEDATA_HPP

#include <cstddef>
#include <png++/png.hpp>

class ImageData
{
public:
    using color_type = png::byte;

    ImageData();
    ImageData(std::string filename);

    std::size_t getWidth() const;
    std::size_t getHeight() const;

    void loadImage(std::string filename);
    const color_type& getPixelRed(std::size_t x, std::size_t y) const;
    const color_type& getPixelGreen(std::size_t x, std::size_t y) const;
    const color_type& getPixelBlue(std::size_t x, std::size_t y) const;
private:
    png::image<png::rgb_pixel> internalImage;

    const png::rgb_pixel& getPixel(std::size_t x, std::size_t y) const;
};

#endif //VCTRSKL_IMAGEDATA_HPP
