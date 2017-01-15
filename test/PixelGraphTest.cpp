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

TEST_F(PixelGraphTest, shouldConstructTwoPointGraphWithNoEdges)
{
    ImageData testedImage("images/chk_1x2.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {0, 0} };

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

TEST_F(PixelGraphTest, shouldConstructAComplete3x3GraphFromUniformImage)
{
    ImageData testedImage("images/uniform_3x3.png");
    testedGraph = new PixelGraph(testedImage);

    std::vector<std::vector<color_type>> expectedResult{ {32+16+8, 2+4+8+16+32, 2+4+8},
                                                         {128+64+32+16+8, 255, 128+1+2+4+8},
                                                         {128+64+32, 2+1+128+64+32, 2+1+128}};

    EXPECT_EQ(expectedResult, testedGraph->getEdgeValues());
}