#include "gtest/gtest.h"
#include "ImageData.hpp"

struct ImageDataTest : testing::Test
{
    using color_type = Color::byte;

    ImageData testedData;

    ImageDataTest()
            : testedData{}
    {}

    virtual ~ImageDataTest()
    {}
};

TEST_F(ImageDataTest, defaultlyConstructedClassShouldHaveNoImage)
{
    EXPECT_EQ(0U, testedData.getWidth());
    EXPECT_EQ(0U, testedData.getHeight());
}

TEST_F(ImageDataTest, zigzagImageShouldBeLoadedAndVerified)
{
    testedData.loadImage("images/chk_2x20.png");
    unsigned int expectedWidth = 20;
    unsigned int extectedHeight = 2;

    EXPECT_EQ(expectedWidth, testedData.getWidth());
    EXPECT_EQ(extectedHeight, testedData.getHeight());

    for(std::size_t i = 0; i < expectedWidth/2; i+=2)
    {
        EXPECT_EQ(255 * (i % 2), testedData.getPixelRed(0, i));
        EXPECT_EQ(255 * (i % 2), testedData.getPixelGreen(0, i));
        EXPECT_EQ(255 * (i % 2), testedData.getPixelBlue(0, i));

        EXPECT_EQ(255 * !(i % 2), testedData.getPixelRed(1, i));
        EXPECT_EQ(255 * !(i % 2), testedData.getPixelGreen(1, i));
        EXPECT_EQ(255 * !(i % 2), testedData.getPixelBlue(1, i));

        EXPECT_EQ(255 * !(i % 2), testedData.getPixelRed(0, i+1));
        EXPECT_EQ(255 * !(i % 2), testedData.getPixelGreen(0, i+1));
        EXPECT_EQ(255 * !(i % 2), testedData.getPixelBlue(0, i+1));

        EXPECT_EQ(255 * (i % 2), testedData.getPixelRed(1, i+1));
        EXPECT_EQ(255 * (i % 2), testedData.getPixelGreen(1, i+1));
        EXPECT_EQ(255 * (i % 2), testedData.getPixelBlue(1, i+1));
    }
}

TEST_F(ImageDataTest, veryDifferentAndVeryCloseColorsShouldBeRecognized)
{
    testedData.loadImage("images/4colors.png");
    unsigned int expectedWidth = 2;
    unsigned int extectedHeight = 2;

    EXPECT_EQ(expectedWidth, testedData.getWidth());
    EXPECT_EQ(extectedHeight, testedData.getHeight());

    EXPECT_EQ(255, testedData.getPixelRed(0, 0));
    EXPECT_EQ(255, testedData.getPixelGreen(0, 0));
    EXPECT_EQ(255, testedData.getPixelBlue(0, 0));

    EXPECT_EQ(0, testedData.getPixelRed(0, 1));
    EXPECT_EQ(0, testedData.getPixelGreen(0, 1));
    EXPECT_EQ(0, testedData.getPixelBlue(0, 1));

    EXPECT_EQ(14, testedData.getPixelRed(1, 0));
    EXPECT_EQ(155, testedData.getPixelGreen(1, 0));
    EXPECT_EQ(254, testedData.getPixelBlue(1, 0));

    EXPECT_EQ(9, testedData.getPixelRed(1, 1));
    EXPECT_EQ(162, testedData.getPixelGreen(1, 1));
    EXPECT_EQ(254, testedData.getPixelBlue(1, 1));
}

TEST_F(ImageDataTest, twoColorImageShouldBeCorrectlyConvertedToYUV)
{
    testedData.loadImage("images/chk_1x2.png");

    color_type extectedWhiteYUV[3] = {255, 128, 128};
    color_type extectedBlackYUV[3] = {0, 128, 128};

    EXPECT_EQ(extectedBlackYUV[0], testedData.getPixelY(0, 0));
    EXPECT_EQ(extectedBlackYUV[1], testedData.getPixelU(0, 0));
    EXPECT_EQ(extectedBlackYUV[2], testedData.getPixelV(0, 0));

    EXPECT_EQ(extectedWhiteYUV[0], testedData.getPixelY(0, 1));
    EXPECT_EQ(extectedWhiteYUV[1], testedData.getPixelU(0, 1));
    EXPECT_EQ(extectedWhiteYUV[2], testedData.getPixelV(0, 1));
}

TEST_F(ImageDataTest, imageWithSimilarColorsShouldBeCorrectlyConvertedToYUV)
{
    ImageData testedData("images/4colors.png");

    color_type expectedY[4] = {255, 0, 124, 126};
    color_type expectedU[4] = {128, 128, 201, 199};
    color_type expectedV[4] = {128, 128, 49, 44};

    for(std::size_t i = 0; i < testedData.getHeight(); ++i)
    for(std::size_t j = 0; j < testedData.getWidth(); ++j)
    {
        EXPECT_EQ(expectedY[j + 2 * i], testedData.getPixelY(i, j));
        EXPECT_EQ(expectedU[j + 2 * i], testedData.getPixelU(i, j));
        EXPECT_EQ(expectedV[j + 2 * i], testedData.getPixelV(i, j));
    }
}