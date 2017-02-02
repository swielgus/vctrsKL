#ifndef VCTRSKL_IMAGEDATA_HPP
#define VCTRSKL_IMAGEDATA_HPP

#include <vector>
#include "Constants.hpp"

class ImageData
{
public:
    using color_type = Color::byte;

    ImageData();
    ImageData(std::string filename);
    ~ImageData();

    void processImage(std::string filename);

    unsigned int getWidth() const;
    unsigned int getHeight() const;

    const color_type& getPixelRed(unsigned int row, unsigned int col) const;
    const color_type& getPixelGreen(unsigned int row, unsigned int col) const;
    const color_type& getPixelBlue(unsigned int row, unsigned int col) const;
    const color_type& getPixelY(unsigned int row, unsigned int col) const;
    const color_type& getPixelU(unsigned int row, unsigned int col) const;
    const color_type& getPixelV(unsigned int row, unsigned int col) const;

    const color_type* getGPUAddressOfYUVColorData() const;
    const int* getGPUAddressOfLabelData() const;

    std::vector< std::vector<int> > getLabelValues() const;
private:
    std::vector<color_type> colorsRGB;
    unsigned int imageWidth;
    unsigned int imageHeight;
    color_type* d_colorYUVData;
    int* d_componentLabels;
    std::vector<color_type> colorsYUV;

    void loadImage(std::string filename);
    void allocateYUVPixelDataOnDevice();
    void createLabelsForSimilarPixels();
    void convertToYUVLocally();
    void setComponentLabelsLocally();

    const color_type& getColorData(const std::vector<color_type>& colors, unsigned int row, unsigned int col,
                                   int colorOffset) const;

    void freeDeviceData();
};

#endif //VCTRSKL_IMAGEDATA_HPP
