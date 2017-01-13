#include "gtest/gtest.h"
#include "ColorConversion.hpp"

struct ColorConversionTest : testing::Test
{
    ColorConversionTest()
    {}

    virtual ~ColorConversionTest()
    {}
};

TEST_F(ColorConversionTest, shouldBeAbleToConvertBlack)
{
    ColorConversion::color_byte color[3] = {0,0,0};
    EXPECT_EQ(0, ColorConversion::convertRGBtoY(color[0],color[1],color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoU(color[0],color[1],color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoV(color[0],color[1],color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertWhite)
{
    ColorConversion::color_byte color[3] = {255,255,255};
    EXPECT_EQ(255, ColorConversion::convertRGBtoY(color[0],color[1],color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoU(color[0],color[1],color[2]));
    EXPECT_EQ(128, ColorConversion::convertRGBtoV(color[0],color[1],color[2]));
}

TEST_F(ColorConversionTest, shouldBeAbleToConvertOrange)
{
    ColorConversion::color_byte color[3] = {230,97,1};
    EXPECT_EQ(125, ColorConversion::convertRGBtoY(color[0],color[1],color[2]));
    EXPECT_EQ(57, ColorConversion::convertRGBtoU(color[0],color[1],color[2]));
    EXPECT_EQ(202, ColorConversion::convertRGBtoV(color[0],color[1],color[2]));
}