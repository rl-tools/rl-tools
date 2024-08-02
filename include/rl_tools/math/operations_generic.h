#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_GENERIC_MATH_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_GENERIC_MATH_H

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::math {

    template<typename T>
    constexpr T PI = 3.141592653589793238462643383279502884L;

    template<typename T>
    constexpr T FRAC_2_SQRTPI = 1.128379167095512573896158903121545172L;
    template<typename T>
    constexpr T SQRT1_2 = 0.707106781186547524400844362104849039L;

    // Implementing sqrt using the Babylonian method (also known as Heron's method)
    template<typename T>
    T sqrt(const devices::math::Generic&, T x) {
        if (x < 0) return -1; // Return -1 for negative numbers to indicate an error
        T guess = x / 2.0;
        T epsilon = 0.00001;
        while ((guess * guess - x) > epsilon || (x - guess * guess) > epsilon) {
            guess = (guess + x / guess) / 2.0;
        }
        return guess;
    }

    template<typename T>
    T exp(const devices::math::Generic&, T x) {
        T sum = 1.0;
        T term = 1.0;
        for (int i = 1; i < 20; ++i) {
            term *= x / i;
            sum += term;
        }
        return sum;
    }

    template<typename T>
    T tanh(const devices::math::Generic&, T x) {
        if (x > 20.0) return 1.0;
        if (x < -20.0) return -1.0;
        T ex = exp(devices::math::Generic(), x);
        T emx = exp(devices::math::Generic(), -x);
        return (ex - emx) / (ex + emx);
    }


    template<typename T>
    T sin(const devices::math::Generic&, T x) {
        T term = x;
        T sum = x;
        for (int i = 1; i < 10; ++i) {
            term *= -x * x / (2 * i * (2 * i + 1));
            sum += term;
        }
        return sum;
    }

    template<typename T>
    T cos(const devices::math::Generic&, T x) {
        T term = 1.0;
        T sum = 1.0;
        for (int i = 1; i < 10; ++i) {
            term *= -x * x / (2 * i * (2 * i - 1));
            sum += term;
        }
        return sum;
    }

    template<typename T>
    T acos(const devices::math::Generic&, T x) {
        if (x < -1 || x > 1) return 0; // acos is only defined for -1 <= x <= 1
        T sum = PI<T>/2;
        T term = x;
        for (int i = 1; i < 10; ++i) {
            term *= (x * x) * (2 * i - 1) / (2 * i);
            sum -= term / (2 * i + 1);
        }
        return sum;
    }

    template<typename TX, typename TY>
    auto pow(const devices::math::Generic&, const TX x, const TY y) {
        if (y == 0) return static_cast<TX>(1);
        if (y < 0) return static_cast<TX>(1) / pow(devices::math::Generic(), x, -y);
        TX temp = pow(devices::math::Generic(), x, y / 2);
        if (static_cast<int>(y) % 2 == 0)
            return temp * temp;
        else
            return x * temp * temp;
    }

    template<typename T>
    T log(const devices::math::Generic&, T x) {
        if (x <= 0) return -1; // log is only defined for positive numbers
        T sum = 0.0;
        T term = (x - 1) / (x + 1);
        T term_squared = term * term;
        for (int i = 0; i < 10; ++i) {
            sum += term / (2 * i + 1);
            term *= term_squared;
        }
        return 2 * sum;
    }

    template<typename T>
    T floor(const devices::math::Generic&, const T x) {
        return static_cast<int>(x);
    }

    template<typename T>
    bool is_nan(const devices::math::Generic&, const T x) {
        return x != x;
    }

    template<typename T>
    bool is_finite(const devices::math::Generic&, const T x) {
        return x <= 1.7976931348623158e+308 && x >= -1.7976931348623158e+308;
    }

    template<typename T>
    T clamp(const devices::math::Generic&, T x, T min, T max) {
        return x < min ? min : (x > max ? max : x);
    }

    template<typename T>
    T min(const devices::math::Generic&, T x, T y) {
        return x < y ? x : y;
    }

    template<typename T>
    T max(const devices::math::Generic&, T x, T y) {
        return x > y ? x : y;
    }

    template<typename T>
    T abs(const devices::math::Generic&, T x) {
        return x > 0 ? x : -x;
    }

    template<typename T>
    T nan(const devices::math::Generic&) {
        return 0.0 / 0.0; // Produces NaN
    }

    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T fast_tanh(const devices::math::Generic& dev, T x){
        x = clamp(dev, x, static_cast<T>(-3.0), static_cast<T>(3.0));
        T x_squared = x * x;
        return x * (27 + x_squared) / (27 + 9 * x_squared);
    }
    template<typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T fast_sigmoid(const devices::math::Generic& dev, T x){
        return (T)0.5 * fast_tanh(dev, (T)0.5 * x) + (T)0.5;
    }


    template<typename T>
    T atan(const devices::math::Generic&, T x) {
        // Constants for polynomial approximation
        T a = 0.9998660;
        T b = -0.3302995;
        T c = 0.1801410;
        T d = -0.0851330;
        T e = 0.0208351;

        // Polynomial approximation for atan(x) in the range [-1, 1]
        T abs_x = x < 0 ? -x : x;
        T x2 = x * x;
        T result = ((a * x2 + b) * x2 + c) * x2 + d;
        result = result * x2 + e;
        result = result * abs_x;

        // Adjust for input sign
        result = x < 0 ? -result : result;

        return result;
    }

    template<typename T>
    T atan2(const devices::math::Generic&, T y, T x) {
        if (x > 0) {
            return atan(y / x);
        } else if (x < 0 && y >= 0) {
            return atan(y / x) + PI<T>;
        } else if (x < 0 && y < 0) {
            return atan(y / x) - PI<T>;
        } else if (x == 0 && y > 0) {
            return PI<T> / 2;
        } else if (x == 0 && y < 0) {
            return -PI<T> / 2;
        }
        return 0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
