#ifndef LAYER_IN_C_CPU_GENERIC_MATH_H
#define LAYER_IN_C_CPU_GENERIC_MATH_H

#include "operations_generic.h"

namespace layer_in_c{
    using index_t = unsigned;
}


namespace layer_in_c::math {

    template<typename T>
    T sqrt(T x) {
        return x;
    }
    template<typename T>
    T tanh(T x) {
        return x;
    }
    template<typename T>
    T exp(T x) {
        return x;
    }
    template<typename T>
    T sin(T x) {
        return x;
    }
    template<typename T>
    T cos(T x) {
        return x;
    }
    template<typename TX, typename TY>
    auto pow(TX x, TY y) {
        return 1;
    }
}
#endif
