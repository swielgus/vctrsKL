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

    bool areYUVColorsSimilar(const color_byte& aY, const color_byte& aU, const color_byte& aV,
                             const color_byte& bY, const color_byte& bU, const color_byte& bV);
}

#endif //VCTRSKL_COLORCONVERSION_HPP
