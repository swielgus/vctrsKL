#include <iostream>
#include <chrono>
#include "CurveOptimizer.hpp"
#include <ImageColorizer.hpp>

int main(int argc, char const *argv[])
{
    std::string fileName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/superMarioWorld2.png";
    if(argc > 1)
        fileName = argv[1];

    auto imageDataStart = std::chrono::steady_clock::now();
    ImageData testedImage(fileName, false);
    auto imageDataEnd = std::chrono::steady_clock::now() - imageDataStart;
    auto imageDataProcessingStart = std::chrono::steady_clock::now();
    testedImage.processImage();
    auto imageDataProcessingEnd = std::chrono::steady_clock::now() - imageDataProcessingStart;
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
    auto imageColorizerStart = std::chrono::steady_clock::now();
    ImageColorizer testedColorizer(testedPolyMap);
    auto imageColorizerEnd = std::chrono::steady_clock::now() - imageColorizerStart;

    std::cout   << "\nTime measured: "
                << "\n ImageData: \t\t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(imageDataEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(imageDataEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(imageDataEnd).count() << " milliseconds"
                << "\n ImageDataProc: \t\t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(imageDataProcessingEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(imageDataProcessingEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(imageDataProcessingEnd).count() << " milliseconds"
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
                << "\n ImageColorizer: \t "
                << std::chrono::duration_cast< std::chrono::nanoseconds >(imageColorizerEnd).count() << " nanoseconds = "
                << std::chrono::duration_cast< std::chrono::microseconds >(imageColorizerEnd).count() << " microseconds = "
                << std::chrono::duration_cast< std::chrono::milliseconds >(imageColorizerEnd).count() << " milliseconds"
              << "\n";

    /*std::cout   << "\n" << std::chrono::duration_cast< std::chrono::milliseconds >(imageDataProcessingEnd).count()
                           + std::chrono::duration_cast< std::chrono::milliseconds >(graphDataEnd).count()
                        << "," << std::chrono::duration_cast< std::chrono::milliseconds >(polygonMapEnd).count()
                        << "," << std::chrono::duration_cast< std::chrono::milliseconds >(curveOptimizerEnd).count()
                               + std::chrono::duration_cast< std::chrono::milliseconds >(imageColorizerEnd).count();*/
    return 0;
}