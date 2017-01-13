#include "ColorConversion.hpp"

ColorConversion::color_byte
ColorConversion::convertRGBtoY(const ColorConversion::color_byte& red,
                               const ColorConversion::color_byte& green,
                               const ColorConversion::color_byte& blue)
{
    //return static_cast<color_byte>(( (  66 * red + 129 * green +  25 * blue + 128) >> 8) +  16);
    return static_cast<color_byte>(0.299 * red + 0.587 * green + 0.114 * blue);
}

ColorConversion::color_byte
ColorConversion::convertRGBtoU(const ColorConversion::color_byte& red,
                               const ColorConversion::color_byte& green,
                               const ColorConversion::color_byte& blue)
{
    //return static_cast<color_byte>(( ( -38 * red -  74 * green + 112 * blue + 128) >> 8) + 128);
    return static_cast<color_byte>((-0.169 * red - 0.331 * green + 0.5 * blue) + 128);
}

ColorConversion::color_byte
ColorConversion::convertRGBtoV(const ColorConversion::color_byte& red,
                               const ColorConversion::color_byte& green,
                               const ColorConversion::color_byte& blue)
{
    //return static_cast<color_byte>(( ( 112 * red -  94 * green -  18 * blue + 128) >> 8) + 128);
    return static_cast<color_byte>((0.5 * red - 0.419 * green - 0.081 * blue) + 128);
}