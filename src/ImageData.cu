#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "ImageData.hpp"
#include "ImageComponentLabeling.hpp"

__global__ void
convertToYUV(ImageData::color_type* inputR, ImageData::color_type* inputG, ImageData::color_type* inputB,
             ImageData::color_type* outputY, ImageData::color_type* outputU, ImageData::color_type* outputV,
             const std::size_t width, const std::size_t height)
{
    int i = threadIdx.x + (blockIdx.x * blockDim.x);
    int j = threadIdx.y + (blockIdx.y * blockDim.y);
    if(i < height && j < width)
    {
        std::size_t idx = j + i * width;

        ImageData::color_type red = inputR[idx];
        ImageData::color_type green = inputG[idx];
        ImageData::color_type blue = inputB[idx];

        outputY[idx] = static_cast<ImageData::color_type>(0.299 * red + 0.587 * green + 0.114 * blue);
        outputU[idx] = static_cast<ImageData::color_type>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
        outputV[idx] = static_cast<ImageData::color_type>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);
    }
}

const ImageData::color_type* ImageData::getGPUAddressOfYColorData() const
{
    return d_colorYData;
}
const ImageData::color_type* ImageData::getGPUAddressOfUColorData() const
{
    return d_colorUData;
}
const ImageData::color_type* ImageData::getGPUAddressOfVColorData() const
{
    return d_colorVData;
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
    allocatePixelDataOnDevice();
    createLabelsForSimilarPixels();
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
        : internalImage{}, d_colorYData{nullptr}, d_colorUData{nullptr}, d_colorVData{nullptr},
          d_componentLabels{nullptr}
{
    this->loadImage(filename);
}

ImageData::ImageData()
        : internalImage{}, d_colorYData{nullptr}, d_colorUData{nullptr}, d_colorVData{nullptr},
          d_componentLabels{nullptr}

{}

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

    color_type* d_inputRData = nullptr;
    color_type* d_inputGData = nullptr;
    color_type* d_inputBData = nullptr;

    cudaMalloc( &d_inputRData, width * height * sizeof(color_type));
    cudaMalloc( &d_inputGData, width * height * sizeof(color_type));
    cudaMalloc( &d_inputBData, width * height * sizeof(color_type));

    int k = 0;
    for(std::size_t i = 0; i < height; ++i)
    for(std::size_t j = 0; j < width; ++j)
    {
        const color_type& currentPixelR = this->getPixelRed(i,j);
        const color_type& currentPixelG = this->getPixelGreen(i,j);
        const color_type& currentPixelB = this->getPixelBlue(i,j);
        cudaMemcpy( d_inputRData+k, &currentPixelR, sizeof(color_type), cudaMemcpyHostToDevice );
        cudaMemcpy( d_inputGData+k, &currentPixelG, sizeof(color_type), cudaMemcpyHostToDevice );
        cudaMemcpy( d_inputBData+k, &currentPixelB, sizeof(color_type), cudaMemcpyHostToDevice );
        ++k;
    }

    cudaMalloc( &d_colorYData, width * height * sizeof(color_type));
    cudaMalloc( &d_colorUData, width * height * sizeof(color_type));
    cudaMalloc( &d_colorVData, width * height * sizeof(color_type));



    dim3 dimBlock(16, 16);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x,
                 (width + dimBlock.y -1)/dimBlock.y);

    convertToYUV<<<dimGrid, dimBlock>>>(d_inputRData, d_inputGData, d_inputBData,
                                        d_colorYData, d_colorUData, d_colorVData, width, height);
    cudaDeviceSynchronize();

    cudaFree(d_inputRData);
    cudaFree(d_inputGData);
    cudaFree(d_inputBData);
}

void ImageData::createLabelsForSimilarPixels()
{
    std::size_t height = this->getHeight();
    std::size_t width = this->getWidth();
    cudaMalloc( &d_componentLabels, width * height * sizeof(int));

    int blockSide = 16;
    dim3 dimBlock(blockSide, blockSide);
    dim3 dimGrid((height + dimBlock.x -1)/dimBlock.x, (width + dimBlock.y -1)/dimBlock.y);

    int numberOfPixelsPerBlock = dimBlock.x * dimBlock.y;

    ImageComponentLabeling::createLocalComponentLabels <<<dimGrid, dimBlock, (numberOfPixelsPerBlock * sizeof(int))+(3 * numberOfPixelsPerBlock * sizeof(Color::byte))>>>(
        d_colorYData, d_colorUData, d_colorVData, d_componentLabels, width, height);
    cudaDeviceSynchronize();

    /*dim3 block(4, 4, blockSide);
    dim3 grid((height + (block.x * blockSide) - 1) / (block.x * blockSide),
              (width + (block.x * blockSide) - 1) / (block.x * blockSide));
    ImageComponentLabeling::mergeSolutionsOnBlockBorders<<<grid, block>>>(
            d_colorYData, d_colorUData, d_colorVData, d_componentLabels, width, height, blockSide);*/

    /*int k =0;
    while(blockSide < width || blockSide < height)
    {
        //compute the number of tiles that are going to be merged in a singe thread block
        int xTiles = 4;
        int yTiles = 4;
        if(xTiles * blockSide > width)
            xTiles = width / blockSide;
        if(yTiles * blockSide > height)
            yTiles = height / blockSide;
        //the number of threads that is going to be used to merge neigboring tiles
        int threadsPerBlock = 32;
        if(blockSide < threadsPerBlock)
            threadsPerBlock = blockSide;
        dim3 block(xTiles, yTiles, threadsPerBlock);
        dim3 grid((height + (block.x * blockSide) - 1) / (block.x * blockSide),
                  (width + (block.x * blockSide) - 1) / (block.x * blockSide));

        //call KERNEL 2


        if(yTiles > xTiles)
            blockSide = yTiles * blockSide;
        else
            blockSide = xTiles * blockSide;

        if(blockSide < width || blockSide < height)
        {
            //update borders (KERNEL 3)
            //cclFlattenEquivalenceTreesAfterMergingTiles(inOutLabelsBuf, inOutLabelsBuf, threadsX, threadsX, imgWidth, imgHeight, dataWidth, log2DataWidth, tileSize);
        }
    }*/
}
