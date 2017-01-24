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

TEST_F(ImageDataTest, constructingComponentLabelsOn1x1ImageShouldGiveOnePoint)
{
    ImageData testedData("images/1x1.png");

    std::vector< std::vector<int> > expectedResult{ {0} };
    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    EXPECT_EQ(expectedResult, actualResult);
}

TEST_F(ImageDataTest, constructingComponentLabelsOn2x2ImageWithSimilatrsShouldGiveThreeLabels)
{
    ImageData testedData("images/4colors.png");

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    EXPECT_TRUE( actualResult[0][0] != actualResult [0][1] );
    EXPECT_TRUE( actualResult[0][0] != actualResult [1][0] );
    EXPECT_TRUE( actualResult[0][1] != actualResult [1][0] );
    EXPECT_TRUE( actualResult[0][0] != actualResult [1][1] );
    EXPECT_TRUE( actualResult[0][1] != actualResult [1][1] );
    EXPECT_TRUE( std::equal(actualResult[1].begin()+1, actualResult[1].end(), actualResult[1].begin()) );
}

TEST_F(ImageDataTest, constructingComponentLabelsOnCloselyButNotTransitivelyRelatedColorsShouldLabelCorrectly)
{
    ImageData testedData("images/closeColorsTransitiveComponent.png");

    std::vector< std::vector<int> > expectedResult{
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,0},
            {0,0,0,0,24},
    };

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    for(int row = 0; row < 5; ++row)
        EXPECT_EQ(expectedResult[row], actualResult[row]);
}

TEST_F(ImageDataTest, constructingComponentLabelsOnBinaryTestImageShouldLabel4Components)
{
    ImageData testedData("images/componentTest.png");

    std::vector< std::vector<int> > expectedResult{
            {0,1,1,1,0,1,1,7},
            {0,0,0,1,0,1,1,7},
            {0,1,1,0,0,1,1,7},
            {0,1,0,0,0,1,1,7},
            {0,1,0,1,1,1,1,1},
            {0,1,0,1,1,1,1,1},
            {0,1,0,1,1,1,54,54},
            {0,1,1,1,1,1,54,54}
    };

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    for(int row = 0; row < 8; ++row)
        EXPECT_EQ(expectedResult[row], actualResult[row]);
}

TEST_F(ImageDataTest, constructingComponentLabelsOnHorizontalLineShouldGiveOneLabelToAllOfThem)
{
    ImageData testedData("images/1x100.png");

    std::vector< std::vector<int> > expectedResult{ {} };
    for(int i = 1; i <= 100; ++i)
        expectedResult[0].push_back(0);

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    EXPECT_EQ(expectedResult, actualResult);
    //EXPECT_TRUE( std::equal(actualResult[0].begin()+1, actualResult[0].end(), actualResult[0].begin()) );
}

TEST_F(ImageDataTest, constructingComponentLabelsOnVerticalLineShouldGiveOneLabelToAllOfThem)
{
    ImageData testedData("images/100x1.png");

    std::vector< std::vector<int> > expectedResult{};
    for(int i = 1; i <= 100; ++i)
        expectedResult.push_back({0});

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    for(int row = 0; row < 100; ++row)
        EXPECT_EQ(expectedResult[row], actualResult[row]);
    //EXPECT_TRUE(std::equal(actualResult.begin()+1, actualResult.end(), actualResult.begin()));
}

TEST_F(ImageDataTest, constructingComponentLabelsOnUnevenImageShouldLabelComponents)
{
    ImageData testedData("images/irregularComponentTest.png");

    std::vector< std::vector<int> > expectedResult{};
    for(int row = 0; row < 51; ++row)
        expectedResult.emplace_back(65,0);
    for(int row = 0; row < 32; ++row)
        for(int col = 0; col < 32; ++col)
        {
            if( (row != col || row >= 16 || col >= 16) && (row != 31-col) )
                expectedResult[row][col] = 1;
        }
    for(int col = 32; col < 65; ++col)
        expectedResult[0][col] = 32;

    std::vector< std::vector<int> > actualResult = testedData.getLabelValues();

    for(int row = 0; row < 51; ++row)
        EXPECT_EQ(expectedResult[row], actualResult[row]);
}