#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ImageData.hpp"


__global__ void
convertToYUV(ImageData::color_type* colorY, ImageData::color_type* colorU, ImageData::color_type* colorV,
             std::size_t width)
{
    int i = threadIdx.x;
    for(int j = 0; j<width; ++j)
    {
        std::size_t idx = j+i*width;

        ImageData::color_type red = colorY[idx];
        ImageData::color_type green = colorU[idx];
        ImageData::color_type blue = colorV[idx];

        colorY[idx] = static_cast<ImageData::color_type>(0.299 * red + 0.587 * green + 0.114 * blue);
        colorU[idx] = static_cast<ImageData::color_type>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
        colorV[idx] = static_cast<ImageData::color_type>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);
    }
}


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
    this->allocatePixelDataOnDevice();
}

const png::rgb_pixel& ImageData::getPixel(std::size_t x, std::size_t y) const
{
    return internalImage.get_row(x).at(y);
}

const ImageData::color_type& ImageData::getPixelRed(std::size_t x, std::size_t y) const
{
    return getPixel(x, y).red;
}

const ImageData::color_type& ImageData::getPixelGreen(std::size_t x, std::size_t y) const
{
    return getPixel(x, y).green;
}

const ImageData::color_type& ImageData::getPixelBlue(std::size_t x, std::size_t y) const
{
    return getPixel(x, y).blue;
}

ImageData::color_type ImageData::getPixelY(std::size_t x, std::size_t y) const
{
    color_type* pixelYValue = new color_type;
    cudaMemcpy( pixelYValue, d_colorYData+(y+x*this->getWidth()), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelYValue;
    delete pixelYValue;
    return result;
}

ImageData::color_type ImageData::getPixelU(std::size_t x, std::size_t y) const
{
    color_type* pixelUValue = new color_type;
    cudaMemcpy( pixelUValue, d_colorUData+(y+x*this->getWidth()), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelUValue;
    delete pixelUValue;
    return result;
}

ImageData::color_type ImageData::getPixelV(std::size_t x, std::size_t y) const
{
    color_type* pixelVValue = new color_type;
    cudaMemcpy( pixelVValue, d_colorVData+(y+x*this->getWidth()), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelVValue;
    delete pixelVValue;
    return result;
}

ImageData::ImageData(std::string filename)
        : internalImage{}, d_colorYData{nullptr}, d_colorUData{nullptr}, d_colorVData{nullptr}
{
    this->loadImage(filename);
}

ImageData::ImageData()
        : internalImage{}, d_colorYData{nullptr}, d_colorUData{nullptr}, d_colorVData{nullptr}
{

}

ImageData::~ImageData()
{
    freeDeviceData();
    //cudaDeviceReset();
}

void ImageData::freeDeviceData()
{
    cudaFree(d_colorYData);
    cudaFree(d_colorUData);
    cudaFree(d_colorVData);
}

void ImageData::allocatePixelDataOnDevice()
{
    freeDeviceData();

    const std::size_t width = this->getWidth();
    const std::size_t height = this->getHeight();
    cudaMalloc( &d_colorYData, width * height * sizeof(color_type));
    cudaMalloc( &d_colorUData, width * height * sizeof(color_type));
    cudaMalloc( &d_colorVData, width * height * sizeof(color_type));

    int k = 0;
    for(std::size_t i = 0; i < height; ++i)
    for(std::size_t j = 0; j < width; ++j)
    {
        const color_type& currentPixelR = this->getPixelRed(i,j);
        const color_type& currentPixelG = this->getPixelGreen(i,j);
        const color_type& currentPixelB = this->getPixelBlue(i,j);
        cudaMemcpy( d_colorYData+k, &currentPixelR, sizeof(color_type), cudaMemcpyHostToDevice );
        cudaMemcpy( d_colorUData+k, &currentPixelG, sizeof(color_type), cudaMemcpyHostToDevice );
        cudaMemcpy( d_colorVData+k, &currentPixelB, sizeof(color_type), cudaMemcpyHostToDevice );
        ++k;
    }

    convertToYUV<<<1,height>>>(d_colorYData, d_colorUData, d_colorVData, width);
}