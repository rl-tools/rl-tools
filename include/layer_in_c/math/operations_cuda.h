#ifndef LAYER_IN_C_MATH_OPERATIONS_CUDA_H
#define LAYER_IN_C_MATH_OPERATIONS_CUDA_H

#ifndef LAYER_IN_C_FUNCTION_PLACEMENT
#define LAYER_IN_C_FUNCTION_PLACEMENT
#endif

#include "operations_generic.h"

#include <layer_in_c/devices/cuda.h>

namespace layer_in_c::math {
    namespace cuda {
        template<typename T>
        constexpr bool check = utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>;
    }

    // CUDA std
    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T sqrt(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::sqrtf(x);
        } else {
            return ::sqrt(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T tanh(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::tanhf(x);
        } else {
            return ::tanh(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T exp(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::expf(x);
        } else {
            return ::exp(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T sin(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::sinf(x);
        } else {
            return ::sin(x);
        }
    }
    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T asin(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::asin(x);
        } else {
            return ::asin(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T cos(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::cosf(x);
        } else {
            return ::cos(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T acos(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::acosf(x);
        } else {
            return ::acos(x);
        }
    }

    template<typename TX, typename TY>
    LAYER_IN_C_FUNCTION_PLACEMENT auto pow(const devices::math::CUDA &, const TX x, const TY y) {
        static_assert(cuda::check<TX>, "CUDA math only supports float and double");
        static_assert(cuda::check<TY>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<TX, double> || utils::typing::is_same_v<TY, double>) {
            return ::pow(x, y);
        } else {
            return ::powf(x, y);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT auto log(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::logf(x);
        } else {
            return ::log(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T floor(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::floorf(x);
        } else {
            return ::floor(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT bool is_nan(const devices::math::CUDA &, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::isnanf(x);
        } else {
            return ::isnan(x);
        }
    }

    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T clamp(const devices::math::CUDA&, const T x, const T min, const T max) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::fmin(max, ::fmax(x, min));
        } else {
            return ::min(max, ::max(x, min));
        }
    }
    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T min(const devices::math::CUDA&, const T a, const T b) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::fmin(a, b);
        } else {
            return ::min(a, b);
        }
    }
    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T max(const devices::math::CUDA&, const T a, const T b) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::fmax(a, b);
        } else {
            return ::max(a, b);
        }
    }
    template<typename T>
    LAYER_IN_C_FUNCTION_PLACEMENT T abs(const devices::math::CUDA&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return ::fabs(x);
        } else {
            return ::abs(x);
        }
    }





//    // CUDA fast
//
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T sqrt(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr(utils::typing::is_same_v<T, float>){
//            return __sqrtf(x);
//        }
//        else{
//            return __sqrt(x);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T tanh(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr(utils::typing::is_same_v<T, float>){
//            return ::tanhf(x);
//        }
//        else{
//            return ::tanh(x);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T exp(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<T, float>) {
//            return __expf(x);
//        }
//        else {
//            return __exp(x);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T sin(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<T, float>){
//            return __sinf(x);
//        }
//        else{
//            return __sin(x);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T cos(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<T, float>){
//            return __cosf(x);
//        }
//        else{
//            return __cos(x);
//        }
//    }
//    template<typename TX, typename TY>
//    LAYER_IN_C_FUNCTION_PLACEMENT auto pow(const devices::math::CUDA_FAST&, const TX x, const TY y) {
//        static_assert(cuda::check<TX>, "CUDA math only supports float and double");
//        static_assert(cuda::check<TY>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<TX, float> && utils::typing::is_same_v<TY, float>){
//            return __powf(x, y);
//        }
//        else{
//            return __pow(x, y);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT auto log(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<T, float>){
//            return __logf(x);
//        }
//        else{
//            return __log(x);
//        }
//    }
//    template<typename T>
//    LAYER_IN_C_FUNCTION_PLACEMENT T floor(const devices::math::CUDA_FAST&, const T x) {
//        static_assert(cuda::check<T>, "CUDA math only supports float and double");
//        if constexpr (utils::typing::is_same_v<T, float>){
//            printf("floor %f %f\n", x, __floorf(x));
//            return __floorf(x);
//        }
//        else{
//            return __floor(x);
//        }
//    }
//}
//
//
}
#endif
