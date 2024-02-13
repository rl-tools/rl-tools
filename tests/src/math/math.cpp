
#include <rl_tools/operations/cpu.h>
#include <gtest/gtest.h>
#include <math.h>
namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;

TEST(RL_TOOLS_MATH, MAIN){
    // if msvc
    #if !defined(_MSC_VER)
    float nan_0 = 0.0f / 0.0f;
    #else
    float nan_0 = NAN;
    #endif

    float nan_1 = NAN;
    float nan_2 = std::numeric_limits<float>::quiet_NaN();
    float nan_3 = std::numeric_limits<float>::signaling_NaN();
    float normal_0 = std::numeric_limits<float>::infinity();
    float normal_1 = std::numeric_limits<float>::epsilon();
    float normal_2 = 0;
    float normal_3 = 1;

    DEVICE device;

#ifndef RL_TOOLS_ENABLE_FAST_MATH
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_0) == std::isnan(nan_0));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_1) == std::isnan(nan_1));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_2) == std::isnan(nan_2));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_3) == std::isnan(nan_3));
    ASSERT_TRUE(rlt::math::is_nan(device.math, normal_0) == std::isnan(normal_0));
    ASSERT_TRUE(rlt::math::is_nan(device.math, normal_1) == std::isnan(normal_1));
    ASSERT_TRUE(rlt::math::is_nan(device.math, normal_2) == std::isnan(normal_2));
    ASSERT_TRUE(rlt::math::is_nan(device.math, normal_3) == std::isnan(normal_3));
#else
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_0));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_1));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_2));
    ASSERT_TRUE(rlt::math::is_nan(device.math, nan_3));
    ASSERT_FALSE(rlt::math::is_nan(device.math, normal_0));
    ASSERT_FALSE(rlt::math::is_nan(device.math, normal_1));
    ASSERT_FALSE(rlt::math::is_nan(device.math, normal_2));
    ASSERT_FALSE(rlt::math::is_nan(device.math, normal_3));
#endif
}
