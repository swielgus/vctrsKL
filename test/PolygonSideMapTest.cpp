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
    graphOfTestedImage.resolveCrossings();
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector<polygon_side_type> expectedResult{ polygon_side_type::Point };

    EXPECT_EQ(expectedResult, testedPolygonMap->getInternalSideTypes());
}

TEST_F(PolygonSideMapTest, shouldConstruct1x100InternalSidesFromHorizontalLineImage)
{
    ImageData testedImage("images/1x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
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
    graphOfTestedImage.resolveCrossings();
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
    graphOfTestedImage.resolveCrossings();
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
    graphOfTestedImage.resolveCrossings();
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

TEST_F(PolygonSideMapTest, shouldConstructTwoRegionBoundariesForBinaryTwoComponentImage)
{
    ImageData testedImage("images/curveHeuTest.png");
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    ClipperLib::Paths expectedResult{
            { {700,100}, {700,200}, {700,300}, {700,400}, {700,500}, {700,600},
              {600,600}, {500,600}, {400,600}, {300,600}, {200,600}, {100,600},
              {0,600}, {0,500}, {0,400}, {0,300}, {0,200}, {0,100}, {0,0}, {100,0},
              {200,0}, {300,0}, {400,0}, {500,0}, {600,0}, {700,0} },
            { {200,300}, {225,375}, {300,400}, {400,400}, {500,400}, {575,425},
              {575,475}, {500,500}, {400,500}, {300,500}, {225,475}, {175,425},
              {125,375}, {100,300}, {125,225}, {175,175}, {225,125}, {300,100},
              {375,125}, {375,175}, {300,200}, {225,225} }
    };

    EXPECT_EQ(expectedResult, testedPolygonMap->getGeneratedRegionBoundaries());
}