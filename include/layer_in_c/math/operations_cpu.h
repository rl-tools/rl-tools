#ifndef LAYER_IN_C_UTILS_GENERIC_MATH_H
#define LAYER_IN_C_UTILS_GENERIC_MATH_H

//#define LAYER_IN_C_USE_CPP_MATH

namespace layer_in_c::utils::math {
#ifdef LAYER_IN_C_USE_CPP_MATH
//    #include <cmath>
#else
//    #include <math.h>
#endif
    template<typename T>
    T clamp(T x, T min, T max){
        return x < min ? min : (x > max ? max : x);
    }

    template<typename T>
    T sqrt(T x) {
//#ifdef LAYER_IN_C_USE_CPP_MATH
//        return std::sqrt(x);
//#else
        std::static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>);
        if constexpr (std::is_same_v<T, float>) {
            return sqrtf(x);
        } else {
            return sqrtf(x);
        }
//#endif
    }
}
#endif
