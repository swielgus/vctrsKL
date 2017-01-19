#ifndef VCTRSKL_CROSSINGRESOLVING_HPP
#define VCTRSKL_CROSSINGRESOLVING_HPP

#include "Constants.hpp"

namespace CrossingResolving
{
    __device__ void doAtomicAnd(Graph::byte* address, Graph::byte value)
    {
        unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t) address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do
        {
            assumed = old;
            min_ = value & (Color::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        }
        while(assumed != old);
    }

    __device__ void doAtomicOr(Graph::byte* address, Graph::byte value)
    {
        unsigned int* base_address = (unsigned int*) ((std::size_t) address & ~3);
        unsigned int selectors[] = {0x3214, 0x3240, 0x3410, 0x4210};
        unsigned int sel = selectors[(size_t) address & 3];
        unsigned int old, assumed, min_, new_;

        old = *base_address;
        do
        {
            assumed = old;
            min_ = value | (Color::byte) __byte_perm(old, 0, ((std::size_t) address & 3) | 0x4440);
            new_ = __byte_perm(old, min_, sel);
            old = atomicCAS(base_address, assumed, new_);
        }
        while(assumed != old);
    }


    __device__ Graph::byte operator+ (GraphEdge a, GraphEdge b)
    {
        return (static_cast<Graph::byte>(a) + static_cast<Graph::byte>(b));
    }

    __device__ Graph::byte operator~(const GraphEdge& a)
    {
        return ~(static_cast<Graph::byte>(a));
    }


    __global__ void removeUnnecessaryCrossings(PixelGraph::edge_type* edges, const std::size_t* dim)
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
                CrossingResolving::doAtomicOr(&edges[idx], GraphEdge::RIGHT + GraphEdge::DOWN);
                CrossingResolving::doAtomicOr(&edges[idx+1], GraphEdge::LEFT + GraphEdge::DOWN);
                CrossingResolving::doAtomicOr(&edges[idx+dim[1]], GraphEdge::UP + GraphEdge::RIGHT);
                CrossingResolving::doAtomicOr(&edges[idx+dim[1]+1], GraphEdge::UP + GraphEdge::LEFT);

                CrossingResolving::doAtomicAnd(&edges[idx], ~GraphEdge::LOWER_RIGHT);
                CrossingResolving::doAtomicAnd(&edges[idx+dim[1]+1], ~GraphEdge::UPPER_LEFT);
                CrossingResolving::doAtomicAnd(&edges[idx+1], ~GraphEdge::LOWER_LEFT);
                CrossingResolving::doAtomicAnd(&edges[idx+dim[1]], ~GraphEdge::UPPER_RIGHT);
            }
        }
    }

    __global__ void resolveCriticalCrossings(PixelGraph::edge_type* edges, const std::size_t* dim)
    {
        int i = threadIdx.x + (blockIdx.x * blockDim.x);
        int j = threadIdx.y + (blockIdx.y * blockDim.y);
        if(i < dim[0]-1 && j < dim[1]-1)
        {
            std::size_t idx = j + i * dim[1];
            bool isThereACrossing = (edges[idx] & static_cast<PixelGraph::edge_type>(GraphEdge::LOWER_RIGHT)) &&
                                    (edges[idx + 1] & static_cast<PixelGraph::edge_type>(GraphEdge::LOWER_LEFT));
        }
    }
}

#endif //VCTRSKL_CROSSINGRESOLVING_HPP
