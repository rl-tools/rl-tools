#ifndef LAYER_IN_C_MATH_OPERATIONS_CPU_H
#define LAYER_IN_C_MATH_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <cmath>

namespace layer_in_c::math {

    template<typename T>
    T sqrt(T x) {
        return std::sqrt(x);
    }
    template<typename T>
    T tanh(T x) {
        return std::tanh(x);
    }
    template<typename T>
    T exp(T x) {
        return std::tanh(x);
    }
    template<typename T>
    T sin(T x) {
        return std::sin(x);
    }
    template<typename T>
    T cos(T x) {
        return std::cos(x);
    }
    template<typename TX, typename TY>
    auto pow(TX x, TY y) {
        return std::pow(x, y);
    }
    template<typename T>
    auto log(T x) {
        return std::log(x);
    }
    template<typename T>
    auto floor(T x) {
        return std::floor(x);
    }
}
#endif
