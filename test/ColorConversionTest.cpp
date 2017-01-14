#include "gtest/gtest.h"
#include "ColorConversion.hpp"
#include "ImageData.hpp"

struct ColorConversionTest : testing::Test
{
    ColorConversionTest()
    {}

    virtual ~ColorConversionTest()
    {}
};

TEST_F(ColorConversionTest, shouldBeAbleToConvertBlack)
{
    ColorConversion::color_byte color[3] = {0, 0, 0};
    EXPECT_EQ(0, ColorConversion::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertWhite)
{
    ColorConversion::color_byte color[3] = {255, 255, 255};
    EXPECT_EQ(255, ColorConversion::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertOrange)
{
    ColorConversion::color_byte color[3] = {230, 97, 1};
    EXPECT_EQ(125, ColorConversion::convertRGBtoY(color[0], color[1], color[2]));
    EXPECT_EQ(57, ColorConversion::convertRGBtoU(color[0], color[1], color[2]));
    EXPECT_EQ(202, ColorConversion::convertRGBtoV(color[0], color[1], color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertLoadedImageColors)
{
    ImageData testedData("images/4colors.png");

    ColorConversion::color_byte expectedY[4] = {255, 0, 124, 126};
    ColorConversion::color_byte expectedU[4] = {128, 128, 201, 199};
    ColorConversion::color_byte expectedV[4] = {128, 128, 49, 44};

    for(std::size_t i = 0; i < testedData.getHeight(); ++i)
        for(std::size_t j = 0; j < testedData.getWidth(); ++j)
        {
            EXPECT_EQ(expectedY[j + 2 * i], ColorConversion::convertRGBtoY(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
            EXPECT_EQ(expectedU[j + 2 * i], ColorConversion::convertRGBtoU(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
            EXPECT_EQ(expectedV[j + 2 * i], ColorConversion::convertRGBtoV(
                    testedData.getPixelRed(i, j),
                    testedData.getPixelGreen(i, j),
                    testedData.getPixelBlue(i, j)));
        }
}

TEST_F(ColorConversionTest, blackAndWhiteShouldNotBeConsideredSimilar)
{
    ColorConversion::color_byte blackYUV[3] = {ColorConversion::convertRGBtoY(0, 0, 0),
                                               ColorConversion::convertRGBtoU(0, 0, 0),
                                               ColorConversion::convertRGBtoV(0, 0, 0)};
    ColorConversion::color_byte whiteYUV[3] = {ColorConversion::convertRGBtoY(255, 255, 255),
                                               ColorConversion::convertRGBtoU(255, 255, 255),
                                               ColorConversion::convertRGBtoV(255, 255, 255)};
    EXPECT_FALSE(ColorConversion::areYUVColorsSimilar(blackYUV[0], blackYUV[1], blackYUV[2],
                                                      whiteYUV[0], whiteYUV[1], whiteYUV[2]));

}

TEST_F(ColorConversionTest, threeShadesOfGrayShouldBeConsideredSimilarPairwiseExceptTheExtremes)
{
    ColorConversion::color_byte lightGrayYUV[3] = {ColorConversion::convertRGBtoY(248, 248, 248),
                                                   ColorConversion::convertRGBtoU(248, 248, 248),
                                                   ColorConversion::convertRGBtoV(248, 248, 248)};
    ColorConversion::color_byte mediumGrayYUV[3] = {ColorConversion::convertRGBtoY(224, 224, 224),
                                                    ColorConversion::convertRGBtoU(224, 224, 224),
                                                    ColorConversion::convertRGBtoV(224, 224, 224)};
    ColorConversion::color_byte darkGrayYUV[3] = {ColorConversion::convertRGBtoY(192, 192, 192),
                                                  ColorConversion::convertRGBtoU(192, 192, 192),
                                                  ColorConversion::convertRGBtoV(192, 192, 192)};

    EXPECT_TRUE(ColorConversion::areYUVColorsSimilar(lightGrayYUV[0], lightGrayYUV[1], lightGrayYUV[2],
                                                     mediumGrayYUV[0], mediumGrayYUV[1], mediumGrayYUV[2]));
    EXPECT_TRUE(ColorConversion::areYUVColorsSimilar(darkGrayYUV[0], darkGrayYUV[1], darkGrayYUV[2],
                                                     mediumGrayYUV[0], mediumGrayYUV[1], mediumGrayYUV[2]));
    EXPECT_FALSE(ColorConversion::areYUVColorsSimilar(lightGrayYUV[0], lightGrayYUV[1], lightGrayYUV[2],
                                                      darkGrayYUV[0], darkGrayYUV[1], darkGrayYUV[2]));
}