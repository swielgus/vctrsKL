#ifndef VCTRSKL_CROSSINGRESOLVING_HPP
#define VCTRSKL_CROSSINGRESOLVING_HPP

#include "Constants.hpp"

namespace CrossingResolving
{
    __device__ void doAtomicAnd(Color::color_byte* address, Color::color_byte value)
    {
        unsigned int *base_address = (unsigned int *)((std::size_t)address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t)address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do {
            assumed = old;
            min_ = value & (Color::color_byte)__byte_perm(old, 0, ((std::size_t)address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        } while (assumed != old);
    }

    __device__ void doAtomicOr(Color::color_byte* address, Color::color_byte value)
    {
        unsigned int *base_address = (unsigned int *)((std::size_t)address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t)address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do {
            assumed = old;
            min_ = value | (Color::color_byte)__byte_perm(old, 0, ((std::size_t)address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        } while (assumed != old);
    }

}

#endif //VCTRSKL_CROSSINGRESOLVING_HPP
