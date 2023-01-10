#ifndef LAYER_IN_C_MATH_OPERATIONS_CUDA_H
#define LAYER_IN_C_MATH_OPERATIONS_CUDA_H

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

#include "operations_generic.h"

#include <layer_in_c/devices/cuda.h>

#include <cmath>

namespace layer_in_c::math {

    template<typename T>
    FUNCTION_PLACEMENT T sqrt(const devices::math::CUDA&, const T x) {
        return std::sqrt(x);
    }
    template<typename T>
    FUNCTION_PLACEMENT T tanh(const devices::math::CUDA&, const T x) {
        return std::tanh(x);
    }
    template<typename T>
    FUNCTION_PLACEMENT T exp(const devices::math::CUDA&, const T x) {
        return std::exp(x);
    }
    template<typename T>
    FUNCTION_PLACEMENT T sin(const devices::math::CUDA&, const T x) {
        return std::sin(x);
    }
    template<typename T>
    FUNCTION_PLACEMENT T cos(const devices::math::CUDA&, const T x) {
        return std::cos(x);
    }
    template<typename TX, typename TY>
    FUNCTION_PLACEMENT auto pow(const devices::math::CUDA&, const TX x, const TY y) {
        return std::pow(x, y);
    }
    template<typename T>
    FUNCTION_PLACEMENT auto log(const devices::math::CUDA&, const T x) {
        return std::log(x);
    }
    template<typename T>
    FUNCTION_PLACEMENT auto floor(const devices::math::CUDA&, const T x) {
        return std::floor(x);
    }
}
#endif
