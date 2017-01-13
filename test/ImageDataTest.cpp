#include "gtest/gtest.h"
#include "ImageData.hpp"

struct ImageDataTest : testing::Test
{
    ImageData testedData;

    ImageDataTest()
            : testedData()
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