#include <iostream>
#include <chrono>
#include <CellMap.hpp>

int main(int argc, char const *argv[])
{
    if(argc < 2)
        return -1;

    std::string outputName = "out.svg";
    if(argc > 2)
        outputName = argv[2];

    auto start = std::chrono::steady_clock::now();
    //ImageData testedImage(argv[1]);
    ImageData testedImage("/home/sw/studia2016-2017Z/pracaMagisterska/conv/rickdan.png");
    PixelGraph graphOfTestedImage(testedImage);
    graphOfTestedImage.resolveCrossings();
    auto duration = std::chrono::steady_clock::now() - start;
    std::cout << "\nTime before cell: " << std::chrono::duration_cast< std::chrono::microseconds >(duration).count() << " microseconds \n"
              << "Time before cell: " << std::chrono::duration_cast< std::chrono::milliseconds >(duration).count() << " milliseconds \n"
              << "Time before cell: " << std::chrono::duration_cast< std::chrono::seconds >(duration).count() << " seconds \n";
    CellMap testedCellMap = CellMap(graphOfTestedImage);

    auto cellPointDetails = testedCellMap.getCellSides();
    auto cellPointPaths = testedCellMap.getCreatedPointPaths();
    const Cell::byte firstThreeBitsMask = 7;
    const int widthOfImage = testedImage.getWidth();
    const int heightOfImage = testedImage.getHeight();

    std::string outputHeader = "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\""
                               + std::to_string(heightOfImage) + "\" width=\"" + std::to_string(widthOfImage) + "\">\n";
    std::string pathDStart = "<path d=\"";
    std::string pathDEnd = " Z\"";

    std::string pathFillStart = " fill=\"rgb(";
    std::string pathFillEnd = ")\"";
    std::string pathEnd = " />";
    std::string outputEnd = "</svg>";

    std::ofstream ofs (outputName, std::ofstream::out);

    ofs << outputHeader;


    for(const auto& path : cellPointPaths)
    {
        std::cout << "\nStart of path\n";
        ofs << pathDStart;
        bool isThePointFirst = true;
        for(const auto& point : path)
        {
            const auto& currentDetails = cellPointDetails[point.idxOfCoordinates];
            std::cout << "-(" << point.idxOfCoordinates << ")E";

            if(isThePointFirst)
            {
                ofs << "M";
                isThePointFirst = false;
            }
            else
            {
                ofs << " L";
            }

            if( (currentDetails.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::Backslash) )
            {
                if(point.directionToCoordinates == GraphEdge::UPPER_LEFT)
                {
                    std::cout << "-Bul[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0];
                }
                else if(point.directionToCoordinates == GraphEdge::UPPER_RIGHT)
                {
                    std::cout << "-Bur[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    std::cout << "-Bur[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0] << " L";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0];
                }
                else if(point.directionToCoordinates == GraphEdge::LOWER_LEFT)
                {
                    std::cout << "-Bll[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    std::cout << "-Bll[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0] << " L";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0];
                }
                else if(point.directionToCoordinates == GraphEdge::LOWER_RIGHT)
                {
                    std::cout << "-Blr[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0];
                }
                else
                {
                    throw std::out_of_range("Unknown position");
                }
            }
            else if( (currentDetails.type & firstThreeBitsMask) == static_cast<Cell::byte>(CellSideType::ForwardSlash) )
            {
                if(point.directionToCoordinates == GraphEdge::UPPER_LEFT)
                {
                    std::cout << "-Ful[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    std::cout << "-Ful[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0] << " L";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0];
                }
                else if(point.directionToCoordinates == GraphEdge::UPPER_RIGHT)
                {
                    std::cout << "-Fur[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0];
                }
                else if(point.directionToCoordinates == GraphEdge::LOWER_LEFT)
                {
                    std::cout << "-Fll[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0];
                }
                else if(point.directionToCoordinates == GraphEdge::LOWER_RIGHT)
                {
                    std::cout << "-Flr[" << currentDetails.pointB[0] << "," << currentDetails.pointB[1] << "]";
                    std::cout << "-Flr[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                    ofs << currentDetails.pointB[1] << "," << currentDetails.pointB[0] << " L";
                    ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0];
                }
                else
                {
                    throw std::out_of_range("Unknown position");
                }
            }
            else
            {
                std::cout << "-P[" << currentDetails.pointA[0] << "," << currentDetails.pointA[1] << "]";
                ofs << currentDetails.pointA[1] << "," << currentDetails.pointA[0];
            }
        }
        ofs << pathDEnd << " fill=\"none\" stroke=\"#000\" stroke-width=\"0.1\" " << pathEnd;
        std::cout << "\nend of path";
    }

    ofs << outputEnd;
    ofs.close();
    return 0;
}