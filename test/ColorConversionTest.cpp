#include "gtest/gtest.h"
#include "Constants.hpp"
#include "ColorConversion.hpp"
#include "ImageData.hpp"

struct ColorConversionTest : testing::Test
{
    using color_type = Color::color_byte;

    ColorConversionTest()
    {}

    virtual ~ColorConversionTest()
    {}
};

TEST_F(ColorConversionTest, shouldBeAbleToConvertBlack)
{
    color_type color[3] = {0, 0, 0};
    EXPECT_EQ(0, ColorOperations::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorOperations::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorOperations::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertWhite)
{
    color_type color[3] = {255, 255, 255};
    EXPECT_EQ(255, ColorOperations::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorOperations::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorOperations::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertOrange)
{
    color_type color[3] = {230, 97, 1};
    EXPECT_EQ(125, ColorOperations::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(57, ColorOperations::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(202, ColorOperations::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertLoadedImageColors)
{
    ImageData testedData("images/4colors.png");

    color_type expectedY[4] = {255, 0, 124, 126};
    color_type expectedU[4] = {128, 128, 201, 199};
    color_type expectedV[4] = {128, 128, 49, 44};

    for(std::size_t i = 0; i < testedData.getHeight(); ++i)
        for(std::size_t j = 0; j < testedData.getWidth(); ++j)
        {
            EXPECT_EQ(expectedY[j + 2 * i], ColorOperations::convertRGBtoY(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
            EXPECT_EQ(expectedU[j + 2 * i], ColorOperations::convertRGBtoU(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
            EXPECT_EQ(expectedV[j + 2 * i], ColorOperations::convertRGBtoV(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
        }
}