#include "gtest/gtest.h"
#include "CellMap.hpp"

struct CellMapTest : testing::Test
{
    using cell_type = CellSideType;

    CellMap* testedCellMap;

    CellMapTest()
            : testedCellMap{nullptr}
    {}

    virtual ~CellMapTest()
    {
        delete testedCellMap;
    }
};

::std::ostream& operator<<(::std::ostream& os, const CellSideType& element)
{
    return os << +static_cast<Cell::byte>(element);  // whatever needed to print bar to os
}

TEST_F(CellMapTest, shouldConstruct1CellFrom1x1Image)
{
    ImageData testedImage("images/1x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ { CellSideType::Point, CellSideType::Point },
                                                          { CellSideType::Point, CellSideType::Point } };

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());
}

TEST_F(CellMapTest, shouldConstruct100SquareCellsFromHorizontalLineImage)
{
    ImageData testedImage("images/1x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ {}, {} };
    cell_type expectedCellValue = CellSideType::Point;

    for(int v = 0; v < 2; ++v)
    {
        for(int i = 0; i < 101; ++i)
        {
            expectedResult[v].push_back(expectedCellValue);
        }
    }

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());
}

TEST_F(CellMapTest, shouldConstruct100SquareCellsFromVerticalLineImage)
{
    ImageData testedImage("images/100x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ };
    std::vector<cell_type> expectedCellRowValues = {CellSideType::Point, CellSideType::Point};

    for(int i = 0; i < 101; ++i)
    {
        expectedResult.push_back(expectedCellRowValues);
    }

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());
}

TEST_F(CellMapTest, shouldConstruct200SquareCellsFromHorizontalDoubleLineImage)
{
    ImageData testedImage("images/bp_2x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ {}, {}, {} };
    cell_type expectedCellValue = CellSideType::Point;
    for(int i = 0; i < 101; ++i)
    {
        expectedResult[0].push_back(expectedCellValue);
        expectedResult[1].push_back(expectedCellValue);
        expectedResult[2].push_back(expectedCellValue);
    }

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());
}

TEST_F(CellMapTest, shouldConstructManySquaresOnTransitiveColorsExceptOneCorner)
{
    ImageData testedImage("images/closeColorsTransitiveComponent.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{{},{},{},{},{},{}};
    cell_type expectedCellValue = CellSideType::Point;
    for(int i = 0; i < 6; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            expectedResult[i].push_back(expectedCellValue);
        }
    }
    expectedResult[4][4] = CellSideType::Backslash;

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());
}

TEST_F(CellMapTest, DISABLED_shouldConstructTwoRegionBoundariesAtLeastForBinaryImage)
{
    ImageData testedImage("images/curveHeuTest.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    /*std::vector< std::vector<cell_type> > expectedResult{{},{},{},{},{},{}};
    cell_type expectedCellValue = CellSideType::Point;
    for(int i = 0; i < 6; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            expectedResult[i].push_back(expectedCellValue);
        }
    }
    expectedResult[4][4] = CellSideType::Backslash;

    EXPECT_EQ(expectedResult, testedCellMap->getCellTypes());*/
}