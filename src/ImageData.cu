#include <iostream>
#include <lodepng.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ImageData.hpp"
#include "ImageComponentLabeling.hpp"

__global__ void
convertToYUV(ImageData::color_type* inputRGB, ImageData::color_type* outputYUV,
             const unsigned int width, const unsigned int height)
{
    int row = threadIdx.x + (blockIdx.x * blockDim.x);
    int col = threadIdx.y + (blockIdx.y * blockDim.y);
    if(row < height && col < width)
    {
        unsigned int idx = 3*col + 3*row * width;

        ImageData::color_type red = inputRGB[idx+0];
        ImageData::color_type green = inputRGB[idx+1];
        ImageData::color_type blue = inputRGB[idx+2];

        outputYUV[idx+0] = static_cast<ImageData::color_type>(0.299 * red + 0.587 * green + 0.114 * blue);
        outputYUV[idx+1] = static_cast<ImageData::color_type>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
        outputYUV[idx+2] = static_cast<ImageData::color_type>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);
    }
}

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
    unsigned error = lodepng::decode(internalImage, imageWidth, imageHeight, filename, LodePNGColorType::LCT_RGB);

    if(error)
        std::cout << "decoder error " << error << ": " << lodepng_error_text(error);
}

const ImageData::color_type& ImageData::getPixelRed(int row, int col) const
{
    int idxOfPixelValues = (3 * col) + (3 * row * getWidth());
    return internalImage[idxOfPixelValues + 0];
}

const ImageData::color_type& ImageData::getPixelGreen(int row, int col) const
{
    int idxOfPixelValues = (3 * col) + (3 * row * getWidth());
    return internalImage[idxOfPixelValues + 1];
}

const ImageData::color_type& ImageData::getPixelBlue(int row, int col) const
{
    int idxOfPixelValues = (3 * col) + (3 * row * getWidth());
    return internalImage[idxOfPixelValues + 2];
}

ImageData::color_type ImageData::getPixelY(unsigned int row, unsigned int col) const
{
    color_type* pixelYValue = new color_type;
    cudaMemcpy( pixelYValue, d_colorYUVData+(3*col+3*row*getWidth() + 0), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelYValue;
    delete pixelYValue;
    return result;
}

ImageData::color_type ImageData::getPixelU(unsigned int row, unsigned int col) const
{
    color_type* pixelUValue = new color_type;
    cudaMemcpy( pixelUValue, d_colorYUVData+(3*col+3*row*getWidth() + 1), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelUValue;
    delete pixelUValue;
    return result;
}

ImageData::color_type ImageData::getPixelV(unsigned int row, unsigned int col) const
{
    color_type* pixelVValue = new color_type;
    cudaMemcpy( pixelVValue, d_colorYUVData+(3*col+3*row*getWidth() + 2), sizeof(color_type), cudaMemcpyDeviceToHost);

    color_type result = *pixelVValue;
    delete pixelVValue;
    return result;
}

void ImageData::processImage(std::string filename)
{
    this->loadImage(filename);
    allocatePixelDataOnDevice();
    createLabelsForSimilarPixels();
}

ImageData::ImageData(std::string filename)
        : internalImage{}, imageWidth{0}, imageHeight{0}, d_colorYUVData{nullptr}, d_componentLabels{nullptr}
{
    processImage(filename);
}

ImageData::ImageData()
        : internalImage{}, imageWidth{0}, imageHeight{0}, d_colorYUVData{nullptr}, d_componentLabels{nullptr}

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

void ImageData::allocatePixelDataOnDevice()
{
    //cudaEvent_t start, stop;
    //cudaEventCreate(&start);
    //cudaEventCreate(&stop);

    freeDeviceData();
    
    std::size_t height = this->getHeight();
    std::size_t width = this->getWidth();

    color_type* d_inputRGBData = nullptr;

    cudaMalloc( &d_inputRGBData, 3 * width * height * sizeof(color_type));
    cudaMemcpy( d_inputRGBData, internalImage.data(), 3 * width * height * sizeof(color_type), cudaMemcpyHostToDevice );

    cudaMalloc( &d_colorYUVData, 3 * width * height * sizeof(color_type));

    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    convertToYUV<<<dimGrid, dimBlock>>>(d_inputRGBData, d_colorYUVData, width, height);
    cudaDeviceSynchronize();

    cudaFree(d_inputRGBData);
}

void ImageData::createLabelsForSimilarPixels()
{
    unsigned int height = this->getHeight();
    unsigned int width = this->getWidth();
    cudaMalloc( &d_componentLabels, width * height * sizeof(int));

    int blockSide = 16;
    dim3 dimBlock(blockSide, blockSide);
    dim3 dimGrid((height + dimBlock.y - 1)/dimBlock.y, (width + dimBlock.x - 1)/dimBlock.x);

    int numberOfPixelsPerBlock = dimBlock.x * dimBlock.y;

    //solve components in local squares
    ImageComponentLabeling::createLocalComponentLabels <<<dimGrid, dimBlock, (numberOfPixelsPerBlock * sizeof(int))+(3 * numberOfPixelsPerBlock * sizeof(Color::byte))>>>(
        d_colorYUVData, d_componentLabels, width, height);
    cudaDeviceSynchronize();

    //merge small results into bigger groups
    while( (blockSide < width || blockSide < height) )
    {
        //compute the number of tiles that are going to be merged in a single thread block
        int numberOfTileRows = 4;
        int numberOfTileCols = 4;

        if(numberOfTileCols * blockSide > width)
            numberOfTileCols = (width + blockSide - 1) / blockSide;

        if(numberOfTileRows * blockSide > height)
            numberOfTileRows = (height + blockSide - 1) / blockSide;

        int threadsPerTile = 32;
        if(blockSide < threadsPerTile)
            threadsPerTile = blockSide;

        dim3 block(numberOfTileRows, numberOfTileCols, threadsPerTile);
        dim3 grid((height + (numberOfTileRows * blockSide) - 1) / (numberOfTileRows * blockSide),
                  (width + (numberOfTileCols * blockSide) - 1) / (numberOfTileCols * blockSide));
        ImageComponentLabeling::mergeSolutionsOnBlockBorders<<<grid, block>>>(
                d_colorYUVData, d_componentLabels, width, height, blockSide);

        if(numberOfTileCols > numberOfTileRows)
            blockSide = numberOfTileCols * blockSide;
        else
            blockSide = numberOfTileRows * blockSide;

        //TODO update labels on borders
    }

    //update all labels
    ImageComponentLabeling::flattenAllEquivalenceTrees<<<dimGrid, dimBlock>>>(d_componentLabels, width, height);
}
