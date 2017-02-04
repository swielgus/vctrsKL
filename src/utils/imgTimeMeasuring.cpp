#include <iostream>
#include <chrono>
#include "CurveOptimizer.hpp"

int main()
{
    std::string fileName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/ftl.png";

    auto imageDataStart = std::chrono::steady_clock::now();
    ImageData testedImage(fileName);
    auto imageDataEnd = std::chrono::steady_clock::now() - imageDataStart;
    auto graphDataStart = std::chrono::steady_clock::now();
    PixelGraph testedGraph(testedImage);
    testedGraph.resolveCrossings();
    auto graphDataEnd = std::chrono::steady_clock::now() - graphDataStart;
    auto polygonMapStart = std::chrono::steady_clock::now();
    PolygonSideMap testedPolyMap(testedGraph);
    auto polygonMapEnd = std::chrono::steady_clock::now() - polygonMapStart;
    auto curveOptimizerStart = std::chrono::steady_clock::now();
    CurveOptimizer testedCurveOptimizer(testedPolyMap);
    auto curveOptimizerEnd = std::chrono::steady_clock::now() - curveOptimizerStart;

    std::cout   << "\nTime measured: "
                << "\n ImageData: \t\t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(imageDataEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(imageDataEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(imageDataEnd).count() << " milliseconds"
                << "\n PixelGraph: \t\t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(graphDataEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(graphDataEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(graphDataEnd).count() << " milliseconds"
                << "\n PolygonSideMap: \t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(polygonMapEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(polygonMapEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(polygonMapEnd).count() << " milliseconds"
                << "\n CurveOptimizer: \t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(curveOptimizerEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(curveOptimizerEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(curveOptimizerEnd).count() << " milliseconds"
              << "\n";
    return 0;
}