#include "gtest/gtest.h"
#include "PixelGraph.hpp"

struct PixelGraphTest : testing::Test
{
    using color_type = Color::byte;

    PixelGraph* testedGraph;

    PixelGraphTest()
            : testedGraph{nullptr}
    {}

    virtual ~PixelGraphTest()
    {
        delete testedGraph;
    }
};

TEST_F(PixelGraphTest, shouldConstruct1x1Graph)
{
    ImageData testedImage("images/1x1.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0} };

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructTwoPointGraphWithNoEdges)
{
    ImageData testedImage("images/chk_1x2.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0} };

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructHorizontalLineGraph)
{
    ImageData testedImage("images/1x100.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {32} };
    for(int i = 1; i <= 98; ++i)
        expectedResult[0].push_back(34);
    expectedResult[0].push_back(2);

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructVerticalLineGraph)
{
    ImageData testedImage("images/100x1.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {8} };
    for(int i = 1; i <= 98; ++i)
        expectedResult.push_back({136});
    expectedResult.push_back({128});

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructAGraphWithFourPointsAndOneEdge)
{
    ImageData testedImage("images/4colors.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0}, {32, 2} };

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructAGraphRepresentingA20Zigzag)
{
    ImageData testedImage("images/chk_2x20.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 4},
            {64, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 65, 1}};

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, shouldConstructAGraphRepresentingA20VerticalZigzag)
{
    ImageData testedImage("images/chk_20x2.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {16,4} };
    for(int i = 1; i <= 18; ++i)
        expectedResult.push_back({80, 5});
    expectedResult.push_back({64,1});

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOn1x1GraphShouldNotChangeAnything)
{
    ImageData testedImage("images/1x1.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0} };
    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnTwoPointGraphWithNoEdgesShouldNotChangeAnything)
{
    ImageData testedImage("images/chk_1x2.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0} };

    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnAGraphWithFourPointsShieldIsolateAllPixels)
{
    ImageData testedImage("images/chk_2x2.png");
    testedGraph = new PixelGraph(testedImage);
    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> expectedResult{ {0, 0}, {0, 0} };

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnAHorizontalLineGraphShouldNotChangeAnything)
{
    ImageData testedImage("images/1x100.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {32} };
    for(int i = 1; i <= 98; ++i)
        expectedResult[0].push_back(34);
    expectedResult[0].push_back(2);

    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnAVerticalLineGraphShouldNotChangeAnything)
{
    ImageData testedImage("images/100x1.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {8} };
    for(int i = 1; i <= 98; ++i)
        expectedResult.push_back({136});
    expectedResult.push_back({128});

    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnFourPointsAndOneEdgeShouldNotChangeAnything)
{
    ImageData testedImage("images/4colors.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0}, {32, 2} };

    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnUniformImageShouldRemoveAllDiagonals)
{
    ImageData testedImage("images/uniform_3x3.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{{40,           42,               10},
                                                        {128 + 32 + 8, 128 + 32 + 8 + 2, 128 + 2 + 8},
                                                        {128 + 32,     2 + 128 + 32,     128 + 2}};
    testedGraph->resolveCrossings();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, constructingGraphOnAComplicatedImageShouldLeaveOnlyCrossColorCrossings)
{
    ImageData testedImage("images/smw_boo_input.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {40,42,42,42,42,42,38,34,34,34,50,42,42,42,42,42,42,10},
            {168,170,170,170,166,210,36,34,34,34,18,165,178,170,170,170,170,138},
            {168,170,170,214,36,66,45,42,42,42,90,33,18,181,170,170,170,138},
            {168,170,214,68,45,106,170,170,170,170,170,43,90,17,181,170,170,138},
            {168,202,72,105,182,170,182,170,170,170,166,178,170,75,9,169,170,138},
            {168,150,132,204,8,201,8,169,170,214,36,18,181,154,144,180,170,138},
            {200,72,105,138,136,136,136,168,202,72,41,74,9,169,91,17,181,138},
            {136,136,168,154,128,156,128,172,154,128,172,138,136,168,170,75,9,137},
            {136,136,168,170,107,170,107,170,170,107,170,138,136,168,170,150,132,140},
            {136,136,168,182,170,182,170,182,170,170,170,154,128,172,218,80,117,138},
            {136,136,216,16,213,20,213,20,181,170,170,170,107,170,170,75,9,137},
            {152,144,180,91,65,93,65,93,1,173,170,170,170,170,170,150,132,140},
            {168,75,9,169,107,170,107,170,107,170,170,170,170,170,202,72,105,138},
            {168,154,144,180,170,170,170,170,170,170,170,170,170,166,146,132,172,138},
            {168,170,91,17,165,178,170,170,170,170,170,166,210,36,66,109,170,138},
            {168,170,170,91,33,18,165,162,162,162,210,36,66,45,106,170,170,138},
            {168,170,170,170,43,90,33,34,34,34,66,45,106,170,170,170,170,138},
            {160,162,162,162,162,162,35,34,34,34,98,162,162,162,162,162,162,130}
    };

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnACurveTestShouldConstructAFinalGraph)
{
    ImageData testedImage("images/curveHeuTest.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {40,42,38,50,42,10},
            {168,198,36,18,177,138},
            {200,72,40,10,9,137},
            {152,128,172,138,136,136},
            {168,107,170,138,136,136},
            {168,170,170,154,128,140},
            {160,162,162,162,99,130}
    };
    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnAnIslandTestShouldConstructAFinalGraph)
{
    ImageData testedImage("images/islandHeuTest.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {40,42,54,10},
            {160,194,4,141},
            {40,74,104,138},
            {160,130,160,130}
    };
    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnASparsePixelsTestShouldConstructAFinalGraph)
{
    ImageData testedImage("images/sparseHeuTest.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {40, 42, 38, 50, 42, 42, 42, 10},
            {168,198,36, 18, 177,170,170,138},
            {200,72, 40, 10, 25, 177,170,138},
            {152,144,160,130,164,19, 177,138},
            {168,27, 49, 74, 40, 10, 9,  137},
            {168,170,27, 145,160,130,132,140},
            {168,170,170,27, 33, 66, 108,138},
            {160,162,162,162,35, 98, 162,130}
    };
    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnASimpleImageShouldMakeAFinalGraph)
{
    ImageData testedImage("images/crossingSign.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {0,44,42,42,38,34,50,42,10},
            {104,170,170,198,36,34,18,177,138},
            {168,170,218,64,44,42,10,9,137},
            {168,170,170,107,170,170,138,136,136},
            {168,166,178,170,170,166,130,132,140},
            {216,32,18,161,194,36,66,108,138},
            {168,43,10,1,68,40,106,170,138},
            {168,170,138,72,16,160,178,170,138},
            {168,170,154,128,28,33,2,173,138},
            {160,162,162,99,162,35,98,162,130}
    };
    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    for(std::size_t row = 0; row < testedImage.getHeight(); ++row)
        EXPECT_EQ(expectedResult[row], result[row]);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnAComplicatedImageShouldGiveAFinalGraph)
{
    ImageData testedImage("images/smw_boo_input.png");
    testedGraph = new PixelGraph(testedImage);
    std::vector<std::vector<color_type>> expectedResult{
            {40, 42, 42, 42, 42, 42, 38,34,34,34,50,42,42,42,42,42,42,10},
            {168,170,170,170,166,194,36,34,34,34,18,165,178,170,170,170,170,138},
            {168,170,170,214,36, 66, 44,42,42,42,90,33,18,181,170,170,170,138},
            {168,170,214,68,45,106,170,170,170,170,170,43,90,17,181,170,170,138},
            {168,202,72,105,182,170,182,170,170,170,166,178,170,75,9,169,170,138},
            {168,150,132,204,8,201,8,169,170,214,36,18,181,154,144,180,170,138},
            {200,72,105,138,136,136,136,168,202,72,41,74,9,169,91,17,181,138},
            {136,136,168,154,128,156,128,172,154,128,172,138,136,168,170,75,9,137},
            {136,136,168,170,107,170,107,170,170,107,170,138,136,168,170,150,132,140},
            {136,136,168,182,170,182,170,182,170,170,170,154,128,172,218,80,117,138},
            {136,136,216,16,213,20,213,20,181,170,170,170,107,170,170,75,9,137},
            {152,144,180,91,65,93,65,93,1,173,170,170,170,170,170,150,132,140},
            {168,75,9,169,107,170,107,170,107,170,170,170,170,170,202,72,105,138},
            {168,154,144,180,170,170,170,170,170,170,170,170,170,166,146,132,172,138},
            {168,170,91,17,165,178,170,170,170,170,170,166,210,36,66,109,170,138},
            {168,170,170,91,33,18,165,162,162,162,210,36,66,45,106,170,170,138},
            {168,170,170,170,43,90,33,34,34,34,66,45,106,170,170,170,170,138},
            {160,162,162,162,162,162,35,34,34,34,98,162,162,162,162,162,162,130}
    };
    expectedResult[1][11] = 161;
    expectedResult[2][3] = 198;
    expectedResult[2][6] = 44;
    expectedResult[2][10] = 26;
    expectedResult[2][13] = 177;
    expectedResult[3][2] = 198;
    expectedResult[3][4] = 44;
    expectedResult[3][12] = 26;
    expectedResult[3][14] = 177;
    expectedResult[4][3] = 104;
    expectedResult[4][13] = 11;
    expectedResult[5][1] = 134;
    expectedResult[5][9] = 198;
    expectedResult[5][12] = 177;
    expectedResult[5][15] = 176;
    expectedResult[6][2] = 104;
    expectedResult[6][10] = 40;
    expectedResult[6][11] = 10;
    expectedResult[6][14] = 27;
    expectedResult[6][16] = 177;
    expectedResult[7][15] = 11;
    expectedResult[8][15] = 134;
    expectedResult[9][16] = 112;
    expectedResult[10][4] = 193;
    expectedResult[10][6] = 193;
    expectedResult[10][8] = 177;
    expectedResult[10][15] = 11;
    expectedResult[11][2] = 176;
    expectedResult[11][3] = 27;
    expectedResult[11][5] = 28;
    expectedResult[11][7] = 28;
    expectedResult[11][15] = 134;
    expectedResult[12][1] = 11;
    expectedResult[12][16] = 104;
    expectedResult[13][3] = 176;
    expectedResult[13][14] = 130;
    expectedResult[14][2] = 27;
    expectedResult[14][4] = 161;
    expectedResult[14][12] = 194;
    expectedResult[14][15] = 108;
    expectedResult[15][3] = 27;
    expectedResult[15][6] = 161;
    expectedResult[15][10] = 194;
    expectedResult[15][13] = 44;
    expectedResult[16][5] = 26;
    expectedResult[16][11] = 44;

    testedGraph->resolveCrossings();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    for(std::size_t row = 0; row < testedImage.getHeight(); ++row)
        EXPECT_EQ(expectedResult[row], result[row]);
}

TEST_F(PixelGraphTest, resolvingCrossingsOnCloselyButNotTransitivelyRelatedColorsShouldGreateAGraphWithOneLonelyPoint)
{
    ImageData testedImage("images/closeColorsTransitiveComponent.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector< std::vector<color_type> > expectedResult{
            {40,42,42,42,10},
            {168,170,170,170,138},
            {168,170,170,170,138},
            {168,170,170,170,134},
            {160,162,162,194,0},
    };

    std::vector< std::vector<color_type> > actualResult = testedGraph->getEdgeValues();

    for(int row = 0; row < 5; ++row)
        EXPECT_EQ(expectedResult[row], actualResult[row]);
}