#ifndef LAYER_IN_C_UTILS_GENERIC_MATH_H
#define LAYER_IN_C_UTILS_GENERIC_MATH_H

#ifndef LAYER_IN_C_FUNCTION_PLACEMENT
#define LAYER_IN_C_FUNCTION_PLACEMENT
#endif

namespace layer_in_c::math {

    template<typename T>
    constexpr T PI = 3.141592653589793238462643383279502884L;

    template<typename T>
    constexpr T FRAC_2_SQRTPI = 1.128379167095512573896158903121545172L;
    template<typename T>
    constexpr T SQRT1_2 = 0.707106781186547524400844362104849039L;

}
#endif
