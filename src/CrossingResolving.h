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


    __global__ void removeUnnecessaryCrossings(PixelGraph::color_type* edges, const std::size_t* dim,
                                                const PixelGraph::color_type* directions)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        if(i < dim[0]-1 && j < dim[1]-1)
        {
            std::size_t idx = j + i * dim[1];

            PixelGraph::color_type upperLeftConnected[12] = {255,48,24,255,56,255,255,255,255,40,40,255};
            PixelGraph::color_type lowerLeftConnected[12] = {96,255,255,192,255,255,224,255,255,160,255,160};
            PixelGraph::color_type upperRightConnected[12] = {6,255,255,12,255,14,255,255,10,255,10,255};
            PixelGraph::color_type lowerRightConnected[12] = {255,3,129,255,255,255,255,131,130,255,255,130};

            int k = 0;
            bool squareIsNotConnected = true;
            while(k < 12 && squareIsNotConnected)
            {
                squareIsNotConnected = !(
                        ((upperLeftConnected[k] == 255) || ((upperLeftConnected[k] & edges[idx]) == upperLeftConnected[k]))
                        && ((upperRightConnected[k] == 255) ||
                            ((upperRightConnected[k] & edges[idx + 1]) == upperRightConnected[k]))
                        && ((lowerLeftConnected[k] == 255) ||
                            ((lowerLeftConnected[k] & edges[idx + dim[1]]) == lowerLeftConnected[k]))
                        && ((lowerRightConnected[k] == 255) ||
                            ((lowerRightConnected[k] & edges[idx + dim[1] + 1]) == lowerRightConnected[k]))
                );
                ++k;
            }

            if(!squareIsNotConnected)
            {
                CrossingResolving::doAtomicOr(&edges[idx], directions[5] + directions[7]);
                CrossingResolving::doAtomicOr(&edges[idx+1], directions[3] + directions[7]);
                CrossingResolving::doAtomicOr(&edges[idx+dim[1]], directions[1] + directions[5]);
                CrossingResolving::doAtomicOr(&edges[idx+dim[1]+1], directions[1] + directions[3]);

                CrossingResolving::doAtomicAnd(&edges[idx], ~directions[8]);
                CrossingResolving::doAtomicAnd(&edges[idx+dim[1]+1], ~directions[0]);
                CrossingResolving::doAtomicAnd(&edges[idx+1], ~directions[6]);
                CrossingResolving::doAtomicAnd(&edges[idx+dim[1]], ~directions[2]);
            }
        }
    }

    __global__ void resolveCriticalCrossings(PixelGraph::color_type* edges, const std::size_t* dim,
                                             const PixelGraph::color_type* directions)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        if(i < dim[0]-1 && j < dim[1]-1)
        {
            std::size_t idx = j + i * dim[1];
            bool isThereACrossing = (edges[idx] & directions[8]) && (edges[idx+1] & directions[6]);
        }
    }
}

#endif //VCTRSKL_CROSSINGRESOLVING_HPP
