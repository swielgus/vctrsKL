#include "ColorConversion.hpp"

ColorOperations::color_type
ColorOperations::convertRGBtoY(const ColorOperations::color_type& red,
                               const ColorOperations::color_type& green,
                               const ColorOperations::color_type& blue)
{
    //return static_cast<color_type>(( (  66 * red + 129 * green +  25 * blue + 128) >> 8) +  16);
    return static_cast<color_type>(0.299 * red + 0.587 * green + 0.114 * blue);
}

ColorOperations::color_type
ColorOperations::convertRGBtoU(const ColorOperations::color_type& red,
                               const ColorOperations::color_type& green,
                               const ColorOperations::color_type& blue)
{
    //return static_cast<color_type>(( ( -38 * red -  74 * green + 112 * blue + 128) >> 8) + 128);
    return static_cast<color_type>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
}

ColorOperations::color_type
ColorOperations::convertRGBtoV(const ColorOperations::color_type& red,
                               const ColorOperations::color_type& green,
                               const ColorOperations::color_type& blue)
{
    //return static_cast<color_type>(( ( 112 * red -  94 * green -  18 * blue + 128) >> 8) + 128);
    return static_cast<color_type>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);
}
