#ifndef VCTRSKL_COLORCONVERSION_HPP
#define VCTRSKL_COLORCONVERSION_HPP

#include "Constants.hpp"

namespace ColorOperations
{
    using color_type = Color::color_byte;

    color_type convertRGBtoY(const color_type& red,
                             const color_type& green,
                             const color_type& blue);
    color_type convertRGBtoU(const color_type& red,
                             const color_type& green,
                             const color_type& blue);
    color_type convertRGBtoV(const color_type& red,
                             const color_type& green,
                             const color_type& blue);
}

#endif //VCTRSKL_COLORCONVERSION_HPP
