#ifndef VCTRSKL_CELLMAP_HPP
#define VCTRSKL_CELLMAP_HPP

#include "PixelGraph.hpp"
#include "Constants.hpp"

class CellMap
{
public:
    using cell_type = Cell::byte;

    CellMap() = delete;
    CellMap(const PixelGraph& graph);
    ~CellMap();

    std::vector< std::vector<cell_type> > getCellValues() const;
private:
    const PixelGraph& sourceGraph;
    cell_type* d_cellData;

    void freeDeviceData();
    void constructPixelCells();
};


#endif //VCTRSKL_CELLMAP_HPP
