#ifndef LAYER_IN_C_MATH_OPERATIONS_CUDA_H
#define LAYER_IN_C_MATH_OPERATIONS_CUDA_H

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

#include "operations_generic.h"

#include <layer_in_c/devices/cuda.h>

#include <cmath>

namespace layer_in_c::math {
    namespace cuda{
        template <typename T>
        constexpr bool check = utils::typing::is_same_v<T, float> || utils::typing::is_same_v<T, double>;
    }

    // CUDA std
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











    // CUDA fast

    template<typename T>
    FUNCTION_PLACEMENT T sqrt(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr(utils::typing::is_same_v<T, float>){
            return __sqrtf(x);
        }
        else{
            return __sqrt(x);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT T tanh(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr(utils::typing::is_same_v<T, float>){
            return __tanhf(x);
        }
        else{
            return __tanh(x);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT T exp(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>) {
            return __expf(x);
        }
        else {
            return __exp(x);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT T sin(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>){
            return __sinf(x);
        }
        else{
            return __sin(x);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT T cos(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>){
            return __cosf(x);
        }
        else{
            return __cos(x);
        }
    }
    template<typename TX, typename TY>
    FUNCTION_PLACEMENT auto pow(const devices::math::CUDA_FAST&, const TX x, const TY y) {
        static_assert(cuda::check<TX>, "CUDA math only supports float and double");
        static_assert(cuda::check<TY>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<TX, float> && utils::typing::is_same_v<TY, float>){
            return __powf(x, y);
        }
        else{
            return __pow(x, y);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT auto log(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>){
            return __logf(x);
        }
        else{
            return __log(x);
        }
    }
    template<typename T>
    FUNCTION_PLACEMENT auto floor(const devices::math::CUDA_FAST&, const T x) {
        static_assert(cuda::check<T>, "CUDA math only supports float and double");
        if constexpr (utils::typing::is_same_v<T, float>){
            return __floorf(x);
        }
        else{
            return __floor(x);
        }
    }
}


#endif
