#include <PolygonSide.hpp>
#include "gtest/gtest.h"
#include "PolygonSideMap.hpp"

struct PolygonSideMapTest : testing::Test
{
    using polygon_side_type = PolygonSide::Type;

    PolygonSideMap* testedPolygonMap;

    PolygonSideMapTest()
            : testedPolygonMap{nullptr}
    {}

    virtual ~PolygonSideMapTest()
    {
        delete testedPolygonMap;
    }
};

TEST_F(PolygonSideMapTest, shouldConstruct1InternalSideFrom1x1Image)
{
    ImageData testedImage("images/1x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ polygon_side_type::Point };

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}

TEST_F(PolygonSideMapTest, shouldConstruct1x100InternalSidesFromHorizontalLineImage)
{
    ImageData testedImage("images/1x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ };
    polygon_side_type expectedCellValue = polygon_side_type::Point;

    for(int i = 0; i < 100; ++i)
    {
        expectedResult.push_back(expectedCellValue);
    }

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}

TEST_F(PolygonSideMapTest, shouldConstruct100x1InternalSidesFromVerticalLineImage)
{
    ImageData testedImage("images/100x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ };
    polygon_side_type expectedCellValue = polygon_side_type::Point;

    for(int i = 0; i < 100; ++i)
    {
        expectedResult.push_back(expectedCellValue);
    }

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}

TEST_F(PolygonSideMapTest, shouldConstruct2x100InternalSidesFromHorizontalDoubleLineImage)
{
    ImageData testedImage("images/bp_2x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ };
    polygon_side_type expectedCellValue = polygon_side_type::Point;
    for(int i = 0; i < 100; ++i)
    {
        for(int v = 0; v < 2; ++v)
        {
            expectedResult.push_back(expectedCellValue);
        }
    }

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}

TEST_F(PolygonSideMapTest, shouldConstructManySquaresOnTransitiveColorsExceptOneCorner)
{
    ImageData testedImage("images/closeColorsTransitiveComponent.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ };
    polygon_side_type expectedCellValue = polygon_side_type::Point;
    for(int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < 5; ++j)
        {
            expectedResult.push_back(expectedCellValue);
        }
    }
    expectedResult[24] = polygon_side_type::Backslash;

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}