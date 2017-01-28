#ifndef VCTRSKL_CELLMAP_HPP
#define VCTRSKL_CELLMAP_HPP

#include "PixelGraph.hpp"
#include "RegionConstructor.hpp"
#include "Constants.hpp"

class CellMap
{
public:
    using cell_type = CellSideType;

    CellMap() = delete;
    CellMap(const PixelGraph& graph);
    ~CellMap();

    std::vector< std::vector<CellSideType> > getCellTypes() const;
private:
    const PixelGraph&  sourceGraph;
    CellSide*          d_cellData;
    RegionConstructor* creatorOfRegions;

    void freeDeviceData();
    void constructPixelCells();
    void createPointPaths();
};


#endif //VCTRSKL_CELLMAP_HPP
