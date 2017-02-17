#include <iostream>
#include <fstream>
#include <popt.h>
#include <CurveOptimizer.hpp>
#include <ImageColorizer.hpp>

unsigned int getCoordinate(const PathPoint& pointData, const std::vector<PolygonSide>& coordinateData,
                           const int coordinateIdx, unsigned int widthOfImage, unsigned int heightOfImage)
{
    if(pointData.rowOfCoordinates == heightOfImage || pointData.colOfCoordinates == widthOfImage)
    {
        if(coordinateIdx == 0)
            return static_cast<unsigned int>(pointData.rowOfCoordinates * 100);
        else
            return static_cast<unsigned int>(pointData.colOfCoordinates * 100);
    }

    const PolygonSide& currentPointData = coordinateData[pointData.colOfCoordinates + pointData.rowOfCoordinates * widthOfImage];

    return (pointData.useBPoint) ? (currentPointData.pointB[coordinateIdx]) : (currentPointData.pointA[coordinateIdx]);
}

bool isPointToBeAControlOne(const PathPoint& pointData, const std::vector<PolygonSide>& coordinateData,
                            unsigned int widthOfImage, unsigned int heightOfImage)
{
    if(pointData.rowOfCoordinates == heightOfImage || pointData.colOfCoordinates == widthOfImage ||
       pointData.rowOfCoordinates == 0 || pointData.colOfCoordinates == 0)
    {
        return false;
    }
    const PolygonSide& currentPointData = coordinateData[pointData.colOfCoordinates + pointData.rowOfCoordinates * widthOfImage];
    int degreeOfPoint = (pointData.useBPoint) ? (currentPointData.getNumberOfRegionsUsingB()) : (currentPointData.getNumberOfRegionsUsingA());
    return (degreeOfPoint < 3);
}

char* outputParameter;
int islandHeuristicMultiplier = 5;
int curveHeuristicMultiplier = 1;
int sparsePixelsMultiplier = 1;
int sparsePixelsRadius = 3;
int omitTJunctionsDuringOptimization = 0;
int doNotPaintGaussCircles = 0;

enum OutputType
{
    REGIONS = 1,
    UNOPTIMIZED_CURVES,
    OPTIMIZED_CURVES
} outputType = OPTIMIZED_CURVES;

poptOption programOptions[] = {
        { "output", 'o', POPT_ARG_STRING, &outputParameter, 0, "Output SVG file location", "FILENAME" },
        {"paint-regions",'v', POPT_ARG_VAL, &outputType, REGIONS, "Paint recognized pixel polygon regions", NULL},
        {"paint-unoptimized", 'u', POPT_ARG_VAL, &outputType, UNOPTIMIZED_CURVES,
         "Paint regions bounded by unoptimized curves", NULL},
        {"paint-optimized",'s', POPT_ARG_VAL, &outputType, OPTIMIZED_CURVES,
         "Paint regions bounded by optimized curves (default)", NULL},
        { "island", 'i', POPT_ARG_INT, &islandHeuristicMultiplier, 0, "Weight of island heuristic (default: 5)", "WEIGHT"},
        { "curves", 'c', POPT_ARG_INT, &curveHeuristicMultiplier, 0, "Multiplier of curve heuristic (default: 1)", "MULTIPLIER"},
        { "sparse-pixels", 'p', POPT_ARG_INT, &sparsePixelsMultiplier, 0, "Multiplier of sparse pixel heuristic (default: 1)",
          "MULTIPLIER"},
        { "sparse-pixels-radius", 'r', POPT_ARG_INT, &sparsePixelsRadius, 0, "Radius of sparse pixel checking window (default: 3)",
          "RADIUS"},
        { "omit-junctions", 'j', 0, &omitTJunctionsDuringOptimization, 0, "Do not optimize T-junctions" },
        { "no-gauss", 'g', 0, &doNotPaintGaussCircles, 0, "Do not paint gaussian blobs" },
        POPT_AUTOHELP
        POPT_TABLEEND
};

int main(int argc, char const *argv[])
{
    if(argc < 2)
    {
        std::cerr << "No file name given!" << std::endl;
        return EXIT_FAILURE;
    }

    poptContext optionsContext = poptGetContext(nullptr, argc, argv, programOptions, 0);
    poptGetNextOpt(optionsContext);
    std::string filename = poptGetArg(optionsContext);
    std::string outputName;

    if((outputParameter == NULL))
        outputName = filename + ".svg";
    else
        outputName = outputParameter;

    ImageData testedImage(filename);
    PixelGraph graphOfTestedImage(testedImage, islandHeuristicMultiplier, curveHeuristicMultiplier,
                                  sparsePixelsMultiplier, sparsePixelsRadius);
    graphOfTestedImage.resolveCrossings();
    PolygonSideMap testedPolyMap(graphOfTestedImage);
    if(outputType != UNOPTIMIZED_CURVES && outputType == OPTIMIZED_CURVES)
        CurveOptimizer testedCurveOptimizer(testedPolyMap, static_cast<bool>(omitTJunctionsDuringOptimization));
    ImageColorizer testedColorizer(testedPolyMap);

    auto coordinateData = testedPolyMap.getInternalSidesFromDevice();
    auto regionBoundaries = testedPolyMap.getPathPointBoundaries();
    auto pointRegionIdxValues = testedColorizer.getPointRegionIdxValues();
    const auto& colorRepresentatives = testedPolyMap.getColorRepresentatives();
    const unsigned int widthOfImage = testedImage.getWidth();
    const unsigned int heightOfImage = testedImage.getHeight();

    std::string outputHeader = "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\""
                               + std::to_string(heightOfImage*100) + "\" width=\"" + std::to_string(widthOfImage*100) + "\">\n";
    std::string pathDStart = "\n<path d=\"";
    std::string pathDEnd = " Z\"";

    std::string defsStart = "\n<defs id=\"defs19878\">";
    std::string defsEnd = "\n</defs>";
    std::string filtersSection = "\n<filter id=\"myGauBlurFil\">\n    <feColorMatrix type=\"matrix\"\n    values=\"1 0 0 0 0\n            0 1 0 0 0\n            0 0 1 0 0\n            0 0 0 1 0 \"/>\n</filter>\n<filter id=\"myGauBlurFilTEMP\"\n    x=\"-50%\" y=\"-50%\" width=\"200%\" height=\"200%\">\n    <feGaussianBlur in=\"SourceGraphic\" stdDeviation=\"50\" id=\"feGaussianBlur7805\" />\n</filter>";
    std::string clipPathStart = "\n    <clipPath id=\"clip";
    std::string clipPathEnd = "\n    </clipPath>";

    std::string pathFillStart = " fill=\"rgb(";
    std::string pathFillEnd = ")\"";
    std::string segmentEnd = " />";

    std::string outputEnd = "</svg>";

    std::vector<std::string> pathTexts;
    pathTexts.resize(regionBoundaries.size());
    for(int i = 0; i < regionBoundaries.size(); ++i)
    {
        const auto& path = regionBoundaries[i];
        std::string thisPathText = pathDStart;
        bool isItTheStartOfPath = true;
        for(int j = 0; j < path.size(); ++j)
        {
            int usedControlPointIdx = (j+1) % path.size();
            int followingControlPointIdx = (j+2) % path.size();

            const PathPoint& precedingControlPoint = path[j];
            const PathPoint& usedControlPoint = path[usedControlPointIdx];
            const PathPoint& followingControlPoint = path[followingControlPointIdx];
            const int precCRow = getCoordinate(precedingControlPoint, coordinateData, 0, widthOfImage, heightOfImage);
            const int precCCol = getCoordinate(precedingControlPoint, coordinateData, 1, widthOfImage, heightOfImage);
            const int curCRow = getCoordinate(usedControlPoint, coordinateData, 0, widthOfImage, heightOfImage);
            const int curCCol = getCoordinate(usedControlPoint, coordinateData, 1, widthOfImage, heightOfImage);
            const int folCRow = getCoordinate(followingControlPoint, coordinateData, 0, widthOfImage, heightOfImage);
            const int folCCol = getCoordinate(followingControlPoint, coordinateData, 1, widthOfImage, heightOfImage);

            double rowOfCurveStartPoint = 0.5 * (precCRow + curCRow);
            double colOfCurveStartPoint = 0.5 * (precCCol + curCCol);
            double rowOfCurveEndPoint = 0.5 * (folCRow + curCRow);
            double colOfCurveEndPoint = 0.5 * (folCCol + curCCol);

            if(isItTheStartOfPath)
            {
                isItTheStartOfPath = false;
                thisPathText += " M";
                thisPathText += std::to_string(colOfCurveStartPoint) + "," + std::to_string(rowOfCurveStartPoint);
            }

            bool doNotPaintCurves = (outputType == REGIONS);

            if( !doNotPaintCurves && isPointToBeAControlOne(usedControlPoint, coordinateData, widthOfImage, heightOfImage) )
            {
                thisPathText += " Q" + std::to_string(curCCol) + "," + std::to_string(curCRow) + " "
                              + std::to_string(colOfCurveEndPoint) + "," + std::to_string(rowOfCurveEndPoint);
            }
            else
            {
                thisPathText += " L"
                    + std::to_string(curCCol) + "," + std::to_string(curCRow) + " L"
                    + std::to_string(colOfCurveEndPoint) + "," + std::to_string(rowOfCurveEndPoint);
            }
        }
        thisPathText += pathDEnd;
        pathTexts[i] = thisPathText;
    }

    std::ofstream ofs (outputName, std::ofstream::out);
    ofs << outputHeader << defsStart << filtersSection;
    for(int i = 0; i < regionBoundaries.size(); ++i)
    {
        ofs << clipPathStart << i << "\">";
        ofs << pathTexts[i] + segmentEnd;
        ofs << clipPathEnd;
    }
    ofs << defsEnd;

    for(int i = 0; i < regionBoundaries.size(); ++i)
    {
        std::string nameOfRegion = "#clip" + std::to_string(i);
        int rowToGetColorFrom = colorRepresentatives[i].X;
        int colToGetColorFrom = colorRepresentatives[i].Y;
        const auto representativeRed = testedImage.getPixelRed(rowToGetColorFrom, colToGetColorFrom);
        const auto representativeGreen = testedImage.getPixelGreen(rowToGetColorFrom, colToGetColorFrom);
        const auto representativeBlue = testedImage.getPixelBlue(rowToGetColorFrom, colToGetColorFrom);

        ofs << pathTexts[i] + " clip-path=\"url(" + nameOfRegion + ")\"" + pathFillStart +
               std::to_string(+representativeRed) + "," + std::to_string(+representativeGreen) + "," +
               std::to_string(+representativeBlue) + pathFillEnd + segmentEnd;

        for(int row = 0; row < heightOfImage; ++row)
        for(int col = 0; col < widthOfImage; ++col)
        {
            const auto& regionIdxOfThisPoint = pointRegionIdxValues[row][col];
            if(i == regionIdxOfThisPoint)
            {
                std::string nameOfRegion = "#clip" + std::to_string(regionIdxOfThisPoint);
                const auto red = testedImage.getPixelRed(row, col);
                const auto green = testedImage.getPixelGreen(row, col);
                const auto blue = testedImage.getPixelBlue(row, col);
                int rowToGetColorFrom = colorRepresentatives[regionIdxOfThisPoint].X;
                int colToGetColorFrom = colorRepresentatives[regionIdxOfThisPoint].Y;
                const auto representativeRed = testedImage.getPixelRed(rowToGetColorFrom, colToGetColorFrom);
                const auto representativeGreen = testedImage.getPixelGreen(rowToGetColorFrom, colToGetColorFrom);
                const auto representativeBlue = testedImage.getPixelBlue(rowToGetColorFrom, colToGetColorFrom);

                if(!doNotPaintGaussCircles)
                {
                    if(red != representativeRed && green != representativeGreen && blue != representativeBlue)
                    {
                        ofs << "\n    <circle style=\"display:inline; filter:url(#myGauBlurFilTEMP); clip-path: url(" << nameOfRegion
                            << ");\"" << pathFillStart << +red
                            << "," << +green << "," << +blue << pathFillEnd
                            << " cx=\"" << col * 100 + 50 << "\" cy=\"" << row * 100 + 50 << "\" r=\"80\" "
                            << " clip-path=\"url(" << nameOfRegion << ")\"" << "/>";
                    }
                }
            }
        }
    }




    /*for(int i = 0; i < regionBoundaries.size(); ++i)
    {
        const auto& path = regionBoundaries[i];

        //ofs << pathDEnd << " fill=\"none\" stroke=\"#000\" stroke-width=\"5\" " << segmentEnd;

        int rowToGetColorFrom = colorRepresentatives[i].X;
        int colToGetColorFrom = colorRepresentatives[i].Y;
        ofs << pathDEnd << pathFillStart << +testedImage.getPixelRed(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelGreen(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelBlue(rowToGetColorFrom, colToGetColorFrom) << pathFillEnd << segmentEnd;
    }*/

    ofs << outputEnd;
    ofs.close();

    poptFreeContext(optionsContext);
    delete[] outputParameter;
    return 0;
}
