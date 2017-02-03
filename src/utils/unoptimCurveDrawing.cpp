#include <iostream>
#include <fstream>
#include <PolygonSideMap.hpp>

int main(int argc, char const *argv[])
{
    std::string filename = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/dolphin.png";
    if(argc > 1)
        filename = argv[1];

    std::string outputName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/utilCreated/curve_out.svg";
    if(argc > 2)
        outputName = argv[2];

    ImageData testedImage(filename);
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
    PolygonSideMap testedPolyMap(graphOfTestedImage);

    auto regionBoundaries = testedPolyMap.getGeneratedRegionBoundaries();
    const auto& colorRepresentatives = testedPolyMap.getColorRepresentatives();
    const int widthOfImage = testedImage.getWidth();
    const int heightOfImage = testedImage.getHeight();

    std::string outputHeader = "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\""
                               + std::to_string(heightOfImage*100) + "\" width=\"" + std::to_string(widthOfImage*100) + "\">\n";
    std::string pathDStart = "<path d=\"";
    std::string pathDEnd = " Z\"";

    std::string pathFillStart = " fill=\"rgb(";
    std::string pathFillEnd = ")\"";
    std::string pathEnd = " />";
    std::string outputEnd = "</svg>";

    std::ofstream ofs (outputName, std::ofstream::out);

    ofs << outputHeader;
    for(int i = 0; i < regionBoundaries.size(); ++i)
    {
        const auto& path = regionBoundaries[i];
        ofs << pathDStart;
        bool isItTheStartOfPath = true;
        int rowOfStartingPoint = path[0].Y;
        int colOfStartingPoint = path[0].X;
        for(int j = 0; j < path.size(); ++j)
        {
            const auto& precedingControlPoint = path[j];
            const auto& usedControlPoint = path[ (j+1) % path.size() ];
            const auto& followingControlPoint = path[ (j+2) % path.size() ];
            int rowOfCurveStartPoint = 0.5 * (precedingControlPoint.Y + usedControlPoint.Y);
            int colOfCurveStartPoint = 0.5 * (precedingControlPoint.X + usedControlPoint.X);
            int rowOfCurveEndPoint = 0.5 * (followingControlPoint.Y + usedControlPoint.Y);
            int colOfCurveEndPoint = 0.5 * (followingControlPoint.X + usedControlPoint.X);

            if(isItTheStartOfPath)
            {
                isItTheStartOfPath = false;
                ofs << " M" << rowOfCurveStartPoint << "," << colOfCurveStartPoint;
            }

            ofs << " Q"
                << usedControlPoint.Y << "," << usedControlPoint.X << " "
                << rowOfCurveEndPoint << "," << colOfCurveEndPoint;
        }
        //ofs << pathDEnd << " fill=\"none\" stroke=\"#000\" stroke-width=\"5\" " << pathEnd;

        int rowToGetColorFrom = colorRepresentatives[i].X;
        int colToGetColorFrom = colorRepresentatives[i].Y;
        ofs << pathDEnd << pathFillStart << +testedImage.getPixelRed(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelGreen(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelBlue(rowToGetColorFrom, colToGetColorFrom) << pathFillEnd << pathEnd;
    }

    ofs << outputEnd;
    ofs.close();
    return 0;
}