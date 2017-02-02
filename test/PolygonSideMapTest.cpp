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

TEST_F(PolygonSideMapTest, shouldTranslateTwoRegionBoundariesForBinaryTwoComponentImage)
{
    ImageData testedImage("images/curveHeuTest.png");
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
    testedPolygonMap = new PolygonSideMap(graphOfTestedImage);

    std::vector< std::vector<PathPoint> > expectedResult{
            { {false,7,1}, {false,7,2}, {false,7,3}, {false,7,4}, {false,7,5}, {false,7,6},
              {false,6,6}, {false,5,6}, {false,4,6}, {false,3,6}, {false,2,6}, {false,1,6},
              {false,0,6}, {false,0,5}, {false,0,4}, {false,0,3}, {false,0,2}, {false,0,1}, {false,0,0}, {false,1,0},
              {false,2,0}, {false,3,0}, {false,4,0}, {false,5,0}, {false,6,0}, {false,7,0} },
            { {false,2,3}, {true,2,4}, {false,3,4}, {false,4,4}, {false,5,4}, {false,6,4},
              {false,6,5}, {false,5,5}, {false,4,5}, {false,3,5}, {true,2,5}, {false,2,4},
              {true,1,4}, {false,1,3}, {true,1,2}, {false,2,2}, {true,2,1}, {false,3,1},
              {false,4,1}, {false,4,2}, {false,3,2}, {true,2,2} }
    };

    auto actualResult = testedPolygonMap->getPathPointBoundaries();

    for(int idxOfPath = 0; idxOfPath < actualResult.size(); ++idxOfPath)
    {
        const auto& currentActualPath = actualResult.at(idxOfPath);
        const auto& currentExpectedPath = expectedResult.at(idxOfPath);
        for(int idxOfElement = 0; idxOfElement < currentActualPath.size(); ++idxOfElement)
        {
            //std::cout << "\n" << idxOfPath << ":" << idxOfElement << "\n";
            EXPECT_EQ(currentExpectedPath.at(idxOfElement).useBPoint, currentActualPath.at(idxOfElement).useBPoint);
            EXPECT_EQ(currentExpectedPath.at(idxOfElement).rowOfCoordinates, currentActualPath.at(idxOfElement).rowOfCoordinates);
            EXPECT_EQ(currentExpectedPath.at(idxOfElement).colOfCoordinates, currentActualPath.at(idxOfElement).colOfCoordinates);
        }
    }
}