#ifndef BACKPROP_TOOLS_MATH_OPERATIONS_ESP32_H
#define BACKPROP_TOOLS_MATH_OPERATIONS_ESP32_H

#include "operations_generic.h"

#include <backprop_tools/devices/esp32.h>

#include <cmath>
#include <algorithm>

namespace backprop_tools::math {

    template<typename T>
    T sqrt(const devices::math::ESP32&, const T x) {
        return std::sqrt(x);
    }
    template<typename T>
    T tanh(const devices::math::ESP32&, const T x) {
        return std::tanh(x);
    }
    template<typename T>
    T exp(const devices::math::ESP32&, const T x) {
        return std::exp(x);
    }
    template<typename T>
    T sin(const devices::math::ESP32&, const T x) {
        return std::sin(x);
    }
    template<typename T>
    T cos(const devices::math::ESP32&, const T x) {
        return std::cos(x);
    }
    template<typename T>
    T acos(const devices::math::ESP32&, const T x) {
        return std::acos(x);
    }
    template<typename TX, typename TY>
    auto pow(const devices::math::ESP32&, const TX x, const TY y) {
        return std::pow(x, y);
    }
    template<typename T>
    auto log(const devices::math::ESP32&, const T x) {
        return std::log(x);
    }
    template<typename T>
    auto floor(const devices::math::ESP32&, const T x) {
        return std::floor(x);
    }
    template<typename T>
    auto is_nan(const devices::math::ESP32&, const T x) {
        return isnan(x);
    }
    template<typename T>
    auto is_finite(const devices::math::ESP32&, const T x) {
        return std::isfinite(x);
    }
    template<typename T>
    T clamp(const devices::math::ESP32&, T x, T min, T max){
        return std::clamp(x, min, max);
    }
    template<typename T>
    T min(const devices::math::ESP32&, T x, T y){
        return std::min(x, y);
    }
    template<typename T>
    T max(const devices::math::ESP32&, T x, T y){
        return std::max(x, y);
    }
    template<typename T>
    T abs(const devices::math::ESP32&, T x){
        return std::abs(x);
    }
    template<typename T>
    T nan(const devices::math::ESP32&){
        return NAN;
    }
}
#endif
