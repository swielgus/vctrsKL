#include <iostream>
#include <fstream>
#include <PolygonSideMap.hpp>

int main(int argc, char const *argv[])
{
    std::string filename = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/dolphin.png";
    if(argc > 1)
        filename = argv[1];

    std::string outputName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/utilCreated/out.svg";
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
        int rowToGetColorFrom = colorRepresentatives[i].X;
        int colToGetColorFrom = colorRepresentatives[i].Y;
        for(int j = 0; j < path.size(); ++j)
        {
            const auto& point = path[j];
            if(isItTheStartOfPath)
            {
                isItTheStartOfPath = false;
                ofs << " M";
            }
            else
            {
                ofs << " L";
            }
            ofs << point.Y << "," << point.X;
        }
        //ofs << pathDEnd << " fill=\"none\" stroke=\"#000\" stroke-width=\"5\" " << pathEnd;

        ofs << pathDEnd << pathFillStart << +testedImage.getPixelRed(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelGreen(rowToGetColorFrom, colToGetColorFrom)
            << "," << +testedImage.getPixelBlue(rowToGetColorFrom, colToGetColorFrom) << pathFillEnd << pathEnd;
    }


        /*const Cell::byte firstThreeBitsMask = 7;
    for(int row = 0; row < heightOfImage; ++row)
        for(int col = 0; col < widthOfImage; ++col)
        {
            int currentIdxInSides = col + row * (widthOfImage+1);

            const auto& detailsOfUpperLeft = cellPointDetails[currentIdxInSides];
            const auto& detailsOfUpperRight = cellPointDetails[currentIdxInSides+1];
            const auto& detailsOfLowerLeft = cellPointDetails[currentIdxInSides+widthOfImage+1];
            const auto& detailsOfLowerRight = cellPointDetails[currentIdxInSides+widthOfImage+2];

            ofs << pathDStart;
            ofs << "M";
            if( (detailsOfUpperLeft.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::Backslash) )
            {
                ofs << detailsOfUpperLeft.pointB[1] << "," << detailsOfUpperLeft.pointB[0];
            }
            else if( (detailsOfUpperLeft.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::ForwardSlash) )
            {
                ofs << detailsOfUpperLeft.pointA[1] << "," << detailsOfUpperLeft.pointA[0] << " L";
                ofs << detailsOfUpperLeft.pointB[1] << "," << detailsOfUpperLeft.pointB[0];
            }
            else
            {
                ofs << detailsOfUpperLeft.pointA[1] << "," << detailsOfUpperLeft.pointA[0];
            }

            ofs << "L";
            if( (detailsOfLowerLeft.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::Backslash) )
            {
                ofs << detailsOfLowerLeft.pointA[1] << "," << detailsOfLowerLeft.pointA[0] << " L";
                ofs << detailsOfLowerLeft.pointB[1] << "," << detailsOfLowerLeft.pointB[0];
            }
            else if( (detailsOfLowerLeft.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::ForwardSlash) )
            {
                ofs << detailsOfLowerLeft.pointA[1] << "," << detailsOfLowerLeft.pointA[0];
            }
            else
            {
                ofs << detailsOfLowerLeft.pointA[1] << "," << detailsOfLowerLeft.pointA[0];
            }

            ofs << "L";
            if( (detailsOfLowerRight.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::Backslash) )
            {
                ofs << detailsOfLowerRight.pointA[1] << "," << detailsOfLowerRight.pointA[0];
            }
            else if( (detailsOfLowerRight.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::ForwardSlash) )
            {
                ofs << detailsOfLowerRight.pointB[1] << "," << detailsOfLowerRight.pointB[0] << " L";
                ofs << detailsOfLowerRight.pointA[1] << "," << detailsOfLowerRight.pointA[0];
            }
            else
            {
                ofs << detailsOfLowerRight.pointA[1] << "," << detailsOfLowerRight.pointA[0];
            }

            ofs << "L";
            if( (detailsOfUpperRight.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::Backslash) )
            {
                ofs << detailsOfUpperRight.pointB[1] << "," << detailsOfUpperRight.pointB[0] << " L";
                ofs << detailsOfUpperRight.pointA[1] << "," << detailsOfUpperRight.pointA[0];
            }
            else if( (detailsOfUpperRight.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::ForwardSlash) )
            {
                ofs << detailsOfUpperRight.pointB[1] << "," << detailsOfUpperRight.pointB[0];
            }
            else
            {
                ofs << detailsOfUpperRight.pointA[1] << "," << detailsOfUpperRight.pointA[0];
            }
            //ofs << pathDEnd << " fill=\"none\" stroke=\"#000\" stroke-width=\"0.1\" " << pathEnd;

            ofs << pathDEnd << pathFillStart << +testedImage.getPixelRed(row, col)
                << "," << +testedImage.getPixelGreen(row, col)
                << "," << +testedImage.getPixelBlue(row, col) << pathFillEnd << pathEnd;
        }*/

    ofs << outputEnd;
    ofs.close();
    return 0;
}