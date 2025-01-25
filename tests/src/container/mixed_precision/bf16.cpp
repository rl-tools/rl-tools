#include <gtest/gtest.h>

class bfloat16{
public:
    std::uint16_t value;
    bfloat16() = default;
    explicit bfloat16(float f){
        std::uint32_t bits;
        std::memcpy(&bits, &f, sizeof(bits));
        std::uint16_t top = static_cast<std::uint16_t>(bits >> 16);
        std::uint16_t lower = static_cast<std::uint16_t>(bits & 0xFFFF);
        if (lower > 0x8000 || (lower == 0x8000 && (top & 1) != 0)) {
            top++;
        }
        value = top;
    }
    explicit operator float() const{
        std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
        float f;
        std::memcpy(&f, &bits, sizeof(f));
        return f;
    }
    bfloat16 operator+(const bfloat16 &other) const{
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        float sum = lhs + rhs;
        return bfloat16(sum);
    }
    bfloat16 operator-(const bfloat16 &other) const{
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        float sum = lhs - rhs;
        return bfloat16(sum);
    }
    bfloat16 operator*(const bfloat16 &other) const{
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        float sum = lhs * rhs;
        return bfloat16(sum);
    }
    bfloat16 operator/(const bfloat16 &other) const{
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        float sum = lhs / rhs;
        return bfloat16(sum);
    }
    bool operator==(const bfloat16 &other) const{
        return (value == other.value);
    }
    bool operator>(const bfloat16 &other) const{
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        return lhs > rhs;
    }
    bool operator<(const bfloat16 &other) const {
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        return lhs < rhs;
    }
    bool operator>=(const bfloat16 &other) const {
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        return lhs >= rhs;
    }
    bool operator<=(const bfloat16 &other) const {
        float lhs = static_cast<float>(*this);
        float rhs = static_cast<float>(other);
        return lhs <= rhs;
    }
    bfloat16& operator+=(const bfloat16 &other){
        *this = *this + other;
        return *this;
    }
    bfloat16& operator-=(const bfloat16 &other){
        *this = *this - other;
        return *this;
    }
    bfloat16& operator*=(const bfloat16 &other){
        *this = *this * other;
        return *this;
    }
    bfloat16& operator/=(const bfloat16 &other){
        *this = *this / other;
        return *this;
    }
};

using _T = __bf16;
using T = bfloat16;
#include <cstdint>
#include <cstring>  // for std::memcpy
#include <cmath>
#include <iostream>

static_assert(sizeof(_T) == 2);
static_assert(sizeof(T) == 2);

bool sign(bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    bool sign = (x_ptr[0] >> 15) & 1;
    return sign;
}

uint16_t exponent(bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    uint16_t exponent_pre = (x_ptr[0] & 0b0111111110000000) >> 7;
    std::cout << "exponent pre: " << std::bitset<8>(exponent_pre) << " value: " << exponent_pre << std::endl;
    int16_t exponent = static_cast<int32_t>(exponent_pre) - 127;
    return exponent;
}
uint16_t mantissa(bfloat16 x) {
    uint16_t* x_ptr = reinterpret_cast<uint16_t*>(&x);
    uint16_t mantissa_pre = x_ptr[0] & 0b0000000001111111;
    std::cout << "mantissa pre: " << std::bitset<7>(mantissa_pre) << " value: " << mantissa_pre + 128 << std::endl;
    uint32_t mantissa = mantissa_pre | 0b10000000;
    return mantissa;
}

TEST(TEST_CONTAINER_MIXED_PRECISION_BF16, MAIN) {
    T a = bfloat16(1000.0), b = bfloat16(0.1);
    _T _a = 1000.0, _b = 0.1;
    uint16_t* a_ptr = reinterpret_cast<uint16_t*>(&a);
    uint16_t* _a_ptr = reinterpret_cast<uint16_t*>(&_a);
    uint16_t* b_ptr = reinterpret_cast<uint16_t*>(&b);
    uint16_t* _b_ptr = reinterpret_cast<uint16_t*>(&_b);
    ASSERT_EQ(std::bitset<16>(*a_ptr), std::bitset<16>(*_a_ptr));
    ASSERT_EQ(std::bitset<16>(*b_ptr), std::bitset<16>(*_b_ptr));
    a += bfloat16(1);
    _a += __bf16(1);
    ASSERT_EQ((float)a, 1000);
    ASSERT_EQ((float)a, (float)_a);
    static_assert(0b1111101000 == 1000);
    ASSERT_EQ(mantissa(a), 0b11111010);
    ASSERT_EQ(sign(a), false);
    ASSERT_EQ(exponent(a), 9);


}