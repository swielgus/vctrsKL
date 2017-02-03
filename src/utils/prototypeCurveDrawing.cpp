#include <iostream>
#include <fstream>
#include <PolygonSideMap.hpp>

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
       (pointData.rowOfCoordinates == 0 && pointData.colOfCoordinates == 0))
    {
        return false;
    }
    const PolygonSide& currentPointData = coordinateData[pointData.colOfCoordinates + pointData.rowOfCoordinates * widthOfImage];
    int degreeOfPoint = (pointData.useBPoint) ? (currentPointData.getNumberOfRegionsUsingB()) : (currentPointData.getNumberOfRegionsUsingA());
    return (degreeOfPoint < 3);
}

int main(int argc, char const *argv[])
{
    std::string filename = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/superMarioWorld.png";
    if(argc > 1)
        filename = argv[1];

    std::string outputName = "/home/sw/studia2016-2017Z/pracaMagisterska/conv/utilCreated/curve_out.svg";
    if(argc > 2)
        outputName = argv[2];

    ImageData testedImage(filename);
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
    PolygonSideMap testedPolyMap(graphOfTestedImage);

    auto coordinateData = testedPolyMap.getInternalSides();
    auto regionBoundaries = testedPolyMap.getPathPointBoundaries();
    const auto& colorRepresentatives = testedPolyMap.getColorRepresentatives();
    const unsigned int widthOfImage = testedImage.getWidth();
    const unsigned int heightOfImage = testedImage.getHeight();

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
                ofs << " M" << colOfCurveStartPoint << "," << rowOfCurveStartPoint;
            }

            if( isPointToBeAControlOne(usedControlPoint, coordinateData, widthOfImage, heightOfImage) )
            {
                ofs << " Q"
                    << curCCol << "," << curCRow << " "
                    << colOfCurveEndPoint << "," << rowOfCurveEndPoint;
            }
            else
            {
                ofs << " L"
                    << curCCol << "," << curCRow << " L"
                    << colOfCurveEndPoint << "," << rowOfCurveEndPoint;
            }

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