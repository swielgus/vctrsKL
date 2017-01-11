//
// Created by sw on 11.01.17.
//

#include "gtest/gtest.h"
#include "ImageData.hpp"

struct ImageDataTest : testing::Test
{
    ImageData testedData;

    ImageDataTest()
        : testedData()
    { }

    virtual ~ImageDataTest()
    { }
};

TEST_F(ImageDataTest, defaultlyConstructedClassShouldHaveNoImage)
{
    EXPECT_EQ(0U, testedData.getWidth());
    EXPECT_EQ(0U, testedData.getHeight());
}