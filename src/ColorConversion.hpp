#ifndef VCTRSKL_COLORCONVERSION_HPP
#define VCTRSKL_COLORCONVERSION_HPP

namespace ColorConversion
{
    using color_byte = unsigned char;

    color_byte convertRGBtoY(const color_byte& red,
                             const color_byte& green,
                             const color_byte& blue);
    color_byte convertRGBtoU(const color_byte& red,
                             const color_byte& green,
                             const color_byte& blue);
    color_byte convertRGBtoV(const color_byte& red,
                             const color_byte& green,
                             const color_byte& blue);
}

#endif //VCTRSKL_COLORCONVERSION_HPP
