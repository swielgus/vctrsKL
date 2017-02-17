#include <iostream>
#include <lodepng.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ImageData.hpp"
#include "ImageComponentIdentifier.hpp"
#include "ImageComponentLabeling.hpp"

const ImageData::color_type* ImageData::getGPUAddressOfYUVColorData() const
{
    return d_colorYUVData;
}
const int* ImageData::getGPUAddressOfLabelData() const
{
    return d_componentLabels;
}

unsigned int ImageData::getWidth() const
{
    return imageWidth;
}

unsigned int ImageData::getHeight() const
{
    return imageHeight;
}

void ImageData::loadImage(std::string filename)
{
    unsigned error = lodepng::decode(colorsRGB, imageWidth, imageHeight, filename, LodePNGColorType::LCT_RGB);

    if(error)
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error);
}

const ImageData::color_type& ImageData::getColorData(const std::vector<ImageData::color_type>& colors,
                                                     unsigned int row, unsigned int col, int colorOffset) const
{
    int idxOfColorValue = (3 * col) + (3 * row * getWidth()) + colorOffset;
    return colors[idxOfColorValue];
}

const ImageData::color_type& ImageData::getPixelRed(unsigned int row, unsigned int col) const
{
    return getColorData(colorsRGB, row, col, 0);
}

const ImageData::color_type& ImageData::getPixelGreen(unsigned int row, unsigned int col) const
{
    return getColorData(colorsRGB, row, col, 1);
}

const ImageData::color_type& ImageData::getPixelBlue(unsigned int row, unsigned int col) const
{
    return getColorData(colorsRGB, row, col, 2);
}

const ImageData::color_type& ImageData::getPixelY(unsigned int row, unsigned int col) const
{
    return getColorData(colorsYUV, row, col, 0);
}

const ImageData::color_type& ImageData::getPixelU(unsigned int row, unsigned int col) const
{
    return getColorData(colorsYUV, row, col, 1);
}

const ImageData::color_type& ImageData::getPixelV(unsigned int row, unsigned int col) const
{
    return getColorData(colorsYUV, row, col, 2);
}

void ImageData::processImage()
{
    convertToYUVLocally();
    allocateYUVPixelDataOnDevice();
    createLabelsForSimilarPixels();
}

void ImageData::processImage(std::string filename)
{
    this->loadImage(filename);
    convertToYUVLocally();
    allocateYUVPixelDataOnDevice();
    createLabelsForSimilarPixels();
}

ImageData::ImageData(std::string filename, bool performImageProcessing)
        : colorsRGB{}, imageWidth{0}, imageHeight{0}, d_colorYUVData{nullptr}, d_componentLabels{nullptr}, colorsYUV{}
{
    if(performImageProcessing)
        processImage(filename);
    else
        this->loadImage(filename);
}

ImageData::ImageData()
        : colorsRGB{}, imageWidth{0}, imageHeight{0}, d_colorYUVData{nullptr}, d_componentLabels{nullptr}, colorsYUV{}

{}

ImageData::~ImageData()
{
    freeDeviceData();
    //cudaDeviceReset();
}

void ImageData::freeDeviceData()
{
    cudaFree(d_colorYUVData);
    cudaFree(d_componentLabels);
}

std::vector< std::vector<int> > ImageData::getLabelValues() const
{
    const std::size_t width = getWidth();
    const std::size_t height = getHeight();

    std::vector< std::vector<int> > result;
    result.resize(height);
    for(std::vector<int>& row : result)
        row.resize(width);

    int* pixelLabels = new int[width * height];
    cudaMemcpy(pixelLabels, d_componentLabels, width * height * sizeof(int), cudaMemcpyDeviceToHost);

    for(std::size_t x = 0; x < height; ++x)
    for(std::size_t y = 0; y < width; ++y)
    {
        result[x][y] = *(pixelLabels + (y + x * width));
    }

    delete[] pixelLabels;

    return result;
}

void ImageData::allocateYUVPixelDataOnDevice()
{
    freeDeviceData();
    
    std::size_t height = this->getHeight();
    std::size_t width = this->getWidth();

    cudaMalloc( &d_colorYUVData, 3 * width * height * sizeof(color_type));
    cudaMemcpy( d_colorYUVData, colorsYUV.data(), 3 * width * height * sizeof(color_type), cudaMemcpyHostToDevice );
}

void ImageData::createLabelsForSimilarPixels()
{
    unsigned int height = this->getHeight();
    unsigned int width = this->getWidth();
    cudaMalloc( &d_componentLabels, width * height * sizeof(int));

    ImageComponentLabeling::setComponentLabels(d_colorYUVData, d_componentLabels, width, height);
    //setComponentLabelsLocally();
}

void ImageData::setComponentLabelsLocally()
{
    const unsigned int width = getWidth();
    const unsigned int height = getHeight();
    ImageComponentIdentifier toolToFindLabels{colorsYUV, width, height};

    const std::vector<int>& completedLabels = toolToFindLabels.getComponentLabels();
    cudaMemcpy( d_componentLabels, completedLabels.data(),
                width * height * sizeof(int), cudaMemcpyHostToDevice );
    cudaDeviceSynchronize();
}

void ImageData::convertToYUVLocally()
{
    colorsYUV.resize(colorsRGB.size());
    const unsigned int height = this->getHeight();
    const unsigned int width = this->getWidth();

    for(int row = 0; row < height; ++row)
    for(int col = 0; col < width; ++col)
    {
        unsigned int idx = 3*col + 3*row * width;

        ImageData::color_type red = colorsRGB[idx+0];
        ImageData::color_type green = colorsRGB[idx+1];
        ImageData::color_type blue = colorsRGB[idx+2];

        colorsYUV[idx+0] = static_cast<ImageData::color_type>(0.299 * red + 0.587 * green + 0.114 * blue);
        colorsYUV[idx+1] = static_cast<ImageData::color_type>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
        colorsYUV[idx+2] = static_cast<ImageData::color_type>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);

        /*colorsYUV[idx+0] = static_cast<ImageData::color_type>((76 * red + 150 * green + 29 * blue + 128) >> 8);
        colorsYUV[idx+1] = static_cast<ImageData::color_type>(((-43 * red - 84 * green + 127 * blue + 128) >> 8) +128);
        colorsYUV[idx+2] = static_cast<ImageData::color_type>(((127 * red - 106 * green - 21 * blue + 128) >> 8) +128);*/
    }
}
