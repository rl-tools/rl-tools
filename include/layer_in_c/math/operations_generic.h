#ifndef LAYER_IN_C_UTILS_GENERIC_MATH_H
#define LAYER_IN_C_UTILS_GENERIC_MATH_H

namespace layer_in_c::utils::math {
    template<typename T>
    T clamp(T x, T min, T max){
        return x < min ? min : (x > max ? max : x);
    }
}
#endif
