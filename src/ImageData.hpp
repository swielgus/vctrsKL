#ifndef VCTRSKL_IMAGEDATA_HPP
#define VCTRSKL_IMAGEDATA_HPP

#include <cstddef>
#include <png++/png.hpp>
#include "Constants.hpp"

class ImageData
{
public:
    using color_type = Color::color_byte;

    ImageData();
    ImageData(std::string filename);
    ~ImageData();

    std::size_t getWidth() const;
    std::size_t getHeight() const;

    void loadImage(std::string filename);
    const color_type& getPixelRed(std::size_t x, std::size_t y) const;
    const color_type& getPixelGreen(std::size_t x, std::size_t y) const;
    const color_type& getPixelBlue(std::size_t x, std::size_t y) const;

    color_type getPixelY(std::size_t x, std::size_t y) const;
    color_type getPixelU(std::size_t x, std::size_t y) const;
    color_type getPixelV(std::size_t x, std::size_t y) const;
private:
    png::image<png::rgb_pixel> internalImage;
    color_type* d_colorYData;
    color_type* d_colorUData;
    color_type* d_colorVData;

    void allocatePixelDataOnDevice();
    void freeDeviceData();
    const png::rgb_pixel& getPixel(std::size_t x, std::size_t y) const;
};

#endif //VCTRSKL_IMAGEDATA_HPP
