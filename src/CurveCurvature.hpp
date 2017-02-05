#ifndef VCTRSKL_CURVECURVATURE_HPP
#define VCTRSKL_CURVECURVATURE_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"
#include <stdio.h>

struct ControlPoint
{
    using cord_type = Polygon::cord_type;
    cord_type row;
    cord_type col;
};

class CurveCurvature
{
public:
    using cord_type = ControlPoint::cord_type;
    using param_type = Curve::param_type;

    __device__ CurveCurvature(const ControlPoint& pointA, const ControlPoint& pointB, const ControlPoint& pointC)
        : controlPointA{pointA}, controlPointB{pointB}, controlPointC{pointC}
    {}

    __device__ cord_type operator()(param_type t) const
    {
        cord_type dividend = abs(
                firstDerivativeOfRow(t) * secondDerivativeOfCol(t) -
                secondDerivativeOfRow(t) * firstDerivativeOfCol(t));
        cord_type divisor = pow(
                firstDerivativeOfRow(t) * firstDerivativeOfRow(t) + firstDerivativeOfCol(t) * firstDerivativeOfCol(t),
                static_cast<cord_type>(3) / static_cast<cord_type>(2));
        return dividend / divisor;
    }

private:
    const ControlPoint& controlPointA;
    const ControlPoint& controlPointB;
    const ControlPoint& controlPointC;

    __device__ cord_type firstDerivativeOfRow(param_type t) const
    {
        return 2 * ((1 - t) * (controlPointB.row - controlPointA.row) + (t) * (controlPointC.row - controlPointB.row));
    }

    __device__ cord_type firstDerivativeOfCol(param_type t) const
    {
        return 2 * ((1 - t) * (controlPointB.col - controlPointA.col) + (t) * (controlPointC.col - controlPointB.col));
    }

    __device__ cord_type secondDerivativeOfRow(param_type t) const
    {
        return 2 * (controlPointA.row - 2 * controlPointB.row + controlPointC.row);
    }

    __device__ cord_type secondDerivativeOfCol(param_type t) const
    {
        return 2 * (controlPointA.col - 2 * controlPointB.col + controlPointC.col);
    }
};

#endif //VCTRSKL_CURVECURVATURE_HPP
