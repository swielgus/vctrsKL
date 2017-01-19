#include "gtest/gtest.h"
#include "PixelGraph.hpp"

struct PixelGraphTest : testing::Test
{
    using color_type = Color::color_byte;

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

TEST_F(PixelGraphTest, shouldConstructAComplete3x3GraphFromUniformImage)
{
    ImageData testedImage("images/uniform_3x3.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {32+16+8, 2+4+8+16+32, 2+4+8},
                                                         {128+64+32+16+8, 255, 128+1+2+4+8},
                                                         {128+64+32, 2+1+128+64+32, 2+1+128}};

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOn1x1GraphShouldNotChangeAnything)
{
    ImageData testedImage("images/1x1.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0} };
    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnTwoPointGraphWithNoEdgesShouldNotChangeAnything)
{
    ImageData testedImage("images/chk_1x2.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0} };

    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
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

    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
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

    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnFourPointsAndOneEdgeShouldNotChangeAnything)
{
    ImageData testedImage("images/4colors.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0}, {32, 2} };

    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingCrossingsOnUniformImageShouldRemoveAllDiagonals)
{
    ImageData testedImage("images/uniform_3x3.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{{40,           42,               10},
                                                        {128 + 32 + 8, 128 + 32 + 8 + 2, 128 + 2 + 8},
                                                        {128 + 32,     2 + 128 + 32,     128 + 2}};
    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();
    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}

TEST_F(PixelGraphTest, resolvingUnusedCrossingsOnAComplicatedImageShouldLeaveOnlyCrossColorCrossings)
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
    testedGraph->resolveUnnecessaryDiagonals();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}

TEST_F(PixelGraphTest, DISABLED_resolvingAllCrossingsOnASimpleImageShouldMakeAFinalGraph)
{
    ImageData testedImage("images/crossingSign.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{
            {0,44,42,42,38,34,50,42,10},
            {104,170,170,198,36,34,18,177,138},
            {104,170,218,64,60,42,10,9,137},
            {104,170,170,107,170,170,138,136,136},
            {168,166,178,170,170,166,130,132,140},
            {216,32,18,161,194,36,66,108,138},
            {168,43,10,1,68,40,106,170,138},
            {104,170,138,72,16,160,178,170,138},
            {104,170,154,128,28,33,2,173,138},
            {160,162,162,99,162,35,98,162,130}
    };
    testedGraph->resolveUnnecessaryDiagonals();
    testedGraph->resolveDisconnectingDiagonals();

    std::vector<std::vector<color_type>> result = testedGraph->getEdgeValues();
    EXPECT_EQ(expectedResult, result);
}