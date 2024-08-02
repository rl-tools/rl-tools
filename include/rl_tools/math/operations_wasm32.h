#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_MATH_OPERATIONS_WASM32_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_MATH_OPERATIONS_WASM32_H

#include "../devices/wasm32.h"

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::math {

    template<typename T>
    T sqrt(const devices::math::WASM32&, const T x) {
        return sqrt(devices::math::Generic{}, x);
    }
    template<typename T>
    T tanh(const devices::math::WASM32&, const T x) {
        return tanh(devices::math::Generic{}, x);
    }
    template<typename T>
    T exp(const devices::math::WASM32&, const T x) {
        return exp(devices::math::Generic{}, x);
    }
    template<typename T>
    T sin(const devices::math::WASM32&, const T x) {
        return sin(devices::math::Generic{}, x);
    }
    template<typename T>
    T cos(const devices::math::WASM32&, const T x) {
        return cos(devices::math::Generic{}, x);
    }
    template<typename T>
    T acos(const devices::math::WASM32&, const T x) {
        return acos(devices::math::Generic{}, x);
    }
    template<typename TX, typename TY>
    TX pow(const devices::math::WASM32&, const TX x, const TY y) {
        return pow(devices::math::Generic{}, x, y);
    }
    template<typename T>
    T log(const devices::math::WASM32&, const T x) {
        return log(devices::math::Generic{}, x);
    }
    template<typename T>
    T floor(const devices::math::WASM32&, const T x) {
        return floor(devices::math::Generic{}, x);
    }
    template<typename T>
    bool is_nan(const devices::math::WASM32&, const T x) {
        return is_nan(devices::math::Generic{}, x);
    }
    template<typename T>
    bool is_finite(const devices::math::WASM32&, const T x) {
        return is_finite(devices::math::Generic{}, x);
    }
    template<typename T>
    T clamp(const devices::math::WASM32&, T x, T min, T max){
        return clamp(devices::math::Generic{}, x, min, max);
    }
    template<typename T>
    T min(const devices::math::WASM32&, T x, T y){
        return min(devices::math::Generic{}, x, y);
    }
    template<typename T>
    T max(const devices::math::WASM32&, T x, T y){
        return max(devices::math::Generic{}, x, y);
    }
    template<typename T>
    T abs(const devices::math::WASM32&, T x){
        return abs(devices::math::Generic{}, x);
    }
    template<typename T>
    T nan(const devices::math::WASM32&){
        return nan<T>(devices::math::Generic{});
    }
    template<typename T>
    T atan2(const devices::math::WASM32&, T a, T b){
        return atan2(devices::math::Generic{}, a, b);
    }
    template<typename T>
    T fast_sigmoid(const devices::math::WASM32& dev, T x) {
        return fast_sigmoid(devices::math::Generic{}, x);
    }
    template<typename T>
    T fast_tanh(const devices::math::WASM32& dev, T x) {
        return fast_tanh(devices::math::Generic{}, x);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
