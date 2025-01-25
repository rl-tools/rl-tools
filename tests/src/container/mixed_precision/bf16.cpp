// #define RL_TOOLS_NAMESPACE_WRAPPER test

#include <rl_tools/numeric_types/bf16.h>
namespace rlt = rl_tools;

#include <gtest/gtest.h>

using _T = __bf16;
using T = rlt::numeric_types::bfloat16;
#include <cstdint>
#include <cstring>  // for std::memcpy
#include <cmath>
#include <iostream>

static_assert(sizeof(_T) == 2);
static_assert(sizeof(T) == 2);

bool sign(rlt::numeric_types::bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    bool sign = (x_ptr[0] >> 15) & 1;
    return sign;
}

uint16_t exponent(rlt::numeric_types::bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    uint16_t exponent_pre = (x_ptr[0] & 0b0111111110000000) >> 7;
    std::cout << "exponent pre: " << std::bitset<8>(exponent_pre) << " value: " << exponent_pre << std::endl;
    int16_t exponent = static_cast<int32_t>(exponent_pre) - 127;
    return exponent;
}
uint16_t mantissa(rlt::numeric_types::bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    uint16_t mantissa_pre = x_ptr[0] & 0b0000000001111111;
    std::cout << "mantissa pre: " << std::bitset<7>(mantissa_pre) << " value: " << mantissa_pre + 128 << std::endl;
    uint32_t mantissa = mantissa_pre | 0b10000000;
    return mantissa;
}

bool compare(rlt::numeric_types::bfloat16 x, __bf16 y) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    uint16_t* y_ptr = reinterpret_cast<uint16_t*>(&y);
    return std::bitset<16>(*x_ptr) == std::bitset<16>(*y_ptr);
}

TEST(TEST_CONTAINER_MIXED_PRECISION_BF16, MAIN) {
    T a = rlt::bfloat16_from_float(1000.0), b = rlt::bfloat16_from_float(0.1);
    _T _a = 1000.0, _b = 0.1;
    a += rlt::bfloat16_from_float(1);
    ASSERT_EQ(rlt::bfloat16_to_float(a), 1000);
    _a += __bf16(1);
    ASSERT_EQ(rlt::bfloat16_to_float(a), (float)_a);
    ASSERT_TRUE(compare(a, _a));
    static_assert(0b1111101000 == 1000);
    ASSERT_EQ(mantissa(a), 0b11111010);
    ASSERT_EQ(sign(a), false);
    ASSERT_EQ(exponent(a), 9);


    ASSERT_TRUE(compare(a + b, _a + _b));
    ASSERT_TRUE(compare(a - b, _a - _b));
    ASSERT_TRUE(compare(a * b, _a * _b));
    ASSERT_TRUE(compare(a / b, _a / _b));


}