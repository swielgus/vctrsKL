#include "gtest/gtest.h"
#include "CellMap.hpp"

struct CellMapTest : testing::Test
{
    using cell_type = Cell::byte;

    CellMap* testedCellMap;

    CellMapTest()
            : testedCellMap{nullptr}
    {}

    virtual ~CellMapTest()
    {
        delete testedCellMap;
    }
};

CellMapTest::cell_type operator+(const CellSide& a, const CellSide& b)
{
    return static_cast<CellMapTest::cell_type>(a) + static_cast<CellMapTest::cell_type>(b);
}
CellMapTest::cell_type operator+(const CellMapTest::cell_type& a, const CellSide& b)
{
    return a + static_cast<CellMapTest::cell_type>(b);
}

TEST_F(CellMapTest, shouldConstruct1CellFrom1x1Image)
{
    ImageData testedImage("images/1x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ { 255 } };

    EXPECT_EQ(expectedResult, testedCellMap->getCellValues());
}

TEST_F(CellMapTest, shouldConstruct100SquareCellsFromHorizontalLineImage)
{
    ImageData testedImage("images/1x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ {} };
    Cell::byte expectedCellValue = CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                   CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C;
    for(int i = 1; i <= 100; ++i)
        expectedResult[0].push_back(expectedCellValue);

    EXPECT_EQ(expectedResult, testedCellMap->getCellValues());
}

TEST_F(CellMapTest, shouldConstruct100SquareCellsFromVerticalLineImage)
{
    ImageData testedImage("images/100x1.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ };
    Cell::byte expectedCellValue = CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                   CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C;
    for(int i = 1; i <= 100; ++i)
        expectedResult.push_back({expectedCellValue});

    EXPECT_EQ(expectedResult, testedCellMap->getCellValues());
}

TEST_F(CellMapTest, shouldConstruct200SquareCellsFromHorizontalDoubleLineImage)
{
    ImageData testedImage("images/bp_2x100.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{ {}, {} };
    Cell::byte expectedCellValue = CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                   CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C;
    for(int i = 1; i <= 100; ++i)
    {
        expectedResult[0].push_back(expectedCellValue);
        expectedResult[1].push_back(expectedCellValue);
    }

    EXPECT_EQ(expectedResult, testedCellMap->getCellValues());
}

TEST_F(CellMapTest, shouldConstructManySquaresOnTransitiveColorsExceptOneCorner)
{
    ImageData testedImage("images/closeColorsTransitiveComponent.png");
    PixelGraph graphOfTestedImage(testedImage);
    testedCellMap = new CellMap(graphOfTestedImage);

    std::vector< std::vector<cell_type> > expectedResult{{},{},{},{},{}};
    Cell::byte expectedSquareValue = CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                     CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C;
    for(int i = 1; i <= 5; ++i)
    {
        expectedResult[0].push_back(expectedSquareValue);
        expectedResult[1].push_back(expectedSquareValue);
        expectedResult[2].push_back(expectedSquareValue);
        if(i < 4)
        {
            expectedResult[3].push_back(expectedSquareValue);
            expectedResult[4].push_back(expectedSquareValue);
        }
    }
    expectedResult[3].push_back(CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                CellSide::LOWER_RIGHT_TYPE_B + CellSide::LOWER_LEFT_TYPE_C);
    expectedResult[3].push_back(CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_C +
                                CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_A);
    expectedResult[4].push_back(CellSide::UPPER_RIGHT_TYPE_A + CellSide::UPPER_LEFT_TYPE_C +
                                CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C);
    expectedResult[4].push_back(CellSide::UPPER_RIGHT_TYPE_C + CellSide::UPPER_LEFT_TYPE_B +
                                CellSide::LOWER_RIGHT_TYPE_C + CellSide::LOWER_LEFT_TYPE_C);

    EXPECT_EQ(expectedResult, testedCellMap->getCellValues());
}