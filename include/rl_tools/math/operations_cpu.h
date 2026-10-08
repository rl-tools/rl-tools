#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_MATH_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_MATH_OPERATIONS_CPU_H

#include "operations_generic.h"

#include "../devices/cpu.h"

#include <cmath>
#include <stdint.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::math {
    union is_nan_struct_float{
        uint32_t i;
        float f;
    };
    union is_nan_struct_double{
        uint64_t i;
        double f;
    };
#ifdef RL_TOOLS_DISABLE_FAST_MATH
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT auto is_nan(const devices::math::CPU&, const T x) {
        return std::isnan(x);
    }
#else
    template <typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool is_nan(const devices::math::CPU, T){
        return false;
    }
    RL_TOOLS_FUNCTION_PLACEMENT inline bool is_nan(const devices::math::CPU, const float x){
        is_nan_struct_float u;
        u.f = x;
        return (u.i & 0x7f800000) == 0x7f800000 && (u.i & 0x007fffff) != 0;
    }
    RL_TOOLS_FUNCTION_PLACEMENT inline bool is_nan(const devices::math::CPU, const double x){
        is_nan_struct_double u;
        u.f = x;
        return (u.i & 0x7ff0000000000000) == 0x7ff0000000000000 && (u.i & 0x000fffffffffffff) != 0;
    }
#endif

    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T sqrt(const devices::math::CPU&, const T x) {
        return std::sqrt(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T cbrt(const devices::math::CPU&, const T x) {
        return std::cbrt(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T tanh(const devices::math::CPU&, const T x) {
        return std::tanh(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T exp(const devices::math::CPU&, const T x) {
        return std::exp(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T sin(const devices::math::CPU&, const T x) {
        return std::sin(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T cos(const devices::math::CPU&, const T x) {
        return std::cos(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T acos(const devices::math::CPU&, const T x) {
        return std::acos(x);
    }
    template<typename TX, typename TY>
    RL_TOOLS_FUNCTION_PLACEMENT constexpr auto pow(const devices::math::CPU&, const TX x, const TY y) {
        return std::pow(x, y);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT auto log(const devices::math::CPU&, const T x) {
        return std::log(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT auto floor(const devices::math::CPU&, const T x) {
        return std::floor(x);
    }

    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT auto is_finite(const devices::math::CPU&, const T x) {
        return std::isfinite(x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T clamp(const devices::math::CPU&, T x, T min, T max){
        return (x < min) ? min : (max < x) ? max : x;
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T min(const devices::math::CPU&, T x, T y){
        return (y < x) ? y : x;
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T max(const devices::math::CPU&, T x, T y){
        return (x < y) ? y : x;
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T abs(const devices::math::CPU&, T x){
        if constexpr (utils::typing::is_same_v<T, float>){
            return std::fabs(x);
        }
        else{
            if constexpr (utils::typing::is_same_v<T, double>){
                return std::abs(x);
            }
            else{
                return (T)std::abs((float)x);
            }
        }
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T nan(const devices::math::CPU&){
        return RL_TOOLS_HOST_ONLY(std::numeric_limits<T>::quiet_NaN());
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T infinity(const devices::math::CPU&){
        return RL_TOOLS_HOST_ONLY(std::numeric_limits<T>::infinity());
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T fast_sigmoid(const devices::math::CPU& dev, T x) {
        return fast_sigmoid(devices::math::Generic{}, x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T fast_tanh(const devices::math::CPU& dev, T x) {
        return fast_tanh(devices::math::Generic{}, x);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T atan2(const devices::math::CPU& dev, T a, T b) {
        return std::atan2(a, b);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
