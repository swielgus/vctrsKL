#ifndef VCTRSKL_CURVECURVATURE_HPP
#define VCTRSKL_CURVECURVATURE_HPP

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "Constants.hpp"
#include <stdio.h>

class CurveCurvature
{
public:
    using cord_type = Polygon::cord_type;
    using param_type = Curve::param_type;

    __device__ CurveCurvature(const cord_type& controlPointARow, const cord_type& controlPointACol,
                              const cord_type& controlPointBRow, const cord_type& controlPointBCol,
                              const cord_type& controlPointCRow, const cord_type& controlPointCCol)
    {
        controlPointA[0] = controlPointARow;
        controlPointA[1] = controlPointACol;
        controlPointB[0] = controlPointBRow;
        controlPointB[1] = controlPointBCol;
        controlPointC[0] = controlPointCRow;
        controlPointC[1] = controlPointCCol;
    }

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
    cord_type controlPointA[2];
    cord_type controlPointB[2];
    cord_type controlPointC[2];

    __device__ cord_type getFirstDerivative(const param_type& t, int coordType) const
    {
        return 2 * ((1 - t) * (controlPointB[coordType] - controlPointA[coordType]) +
                    (t) * (controlPointC[coordType] - controlPointB[coordType]));
    }

    __device__ cord_type firstDerivativeOfRow(param_type t) const
    {
        return getFirstDerivative(t, 0);
    }

    __device__ cord_type firstDerivativeOfCol(param_type t) const
    {
        return getFirstDerivative(t, 1);
    }

    __device__ cord_type getSecondDerivative(const param_type& t, int coordType) const
    {
        return 2 * (controlPointA[coordType] - 2 * controlPointB[coordType] + controlPointC[coordType]);
    }

    __device__ cord_type secondDerivativeOfRow(param_type t) const
    {
        return getSecondDerivative(t, 0);
    }

    __device__ cord_type secondDerivativeOfCol(param_type t) const
    {
        return getSecondDerivative(t, 1);
    }
};

#endif //VCTRSKL_CURVECURVATURE_HPP
