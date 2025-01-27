// #define RL_TOOLS_NAMESPACE_WRAPPER test
#include <rl_tools/numeric_types/bf16.h>
#include <rl_tools/operations/cpu.h>
namespace rlt = rl_tools;
namespace rlt = rl_tools;

#include <gtest/gtest.h>
#include <bitset>

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
    T a = 1000.0, b = 0.1;
    _T _a = __bf16(1000.0), _b = __bf16(0.1);
    a += rlt::numeric_types::bfloat16(1);
    ASSERT_EQ(static_cast<float>(a), 1000);
    _a += __bf16(1);
    ASSERT_EQ(static_cast<float>(a), (float)_a);
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

template <typename DEVICE, typename DEVICE::index_t M, typename DEVICE::index_t N, typename DEVICE::index_t K>
std::tuple<double, double, double> test(DEVICE& device){
    using TI = typename DEVICE::index_t;
    constexpr TI UNIDIM = 128;
    using T_FP32 = float;
    using T_FP64 = double;
    using T_BF16 = rlt::numeric_types::bfloat16;
    typename DEVICE::SPEC::RANDOM::template ENGINE<> rng;
    rlt::Matrix<rlt::matrix::Specification<T_BF16, TI, M, K>> A_BF16;
    rlt::Matrix<rlt::matrix::Specification<T_BF16, TI, K, N>> B_BF16;
    rlt::Matrix<rlt::matrix::Specification<T_BF16, TI, M, N>> C_BF16;
    rlt::Matrix<rlt::matrix::Specification<T_FP32, TI, M, K>> A_FP32;
    rlt::Matrix<rlt::matrix::Specification<T_FP32, TI, K, N>> B_FP32;
    rlt::Matrix<rlt::matrix::Specification<T_FP32, TI, M, N>> C_FP32, C_BF16_FP32_ACC;
    rlt::Matrix<rlt::matrix::Specification<T_FP64, TI, M, K>> A_FP64;
    rlt::Matrix<rlt::matrix::Specification<T_FP64, TI, K, N>> B_FP64;
    rlt::Matrix<rlt::matrix::Specification<T_FP64, TI, M, N>> C_FP64, C_BF16_FP64, C_FP32_FP64;
    rlt::malloc(device, A_BF16);
    rlt::malloc(device, B_BF16);
    rlt::malloc(device, C_BF16);
    rlt::malloc(device, A_FP32);
    rlt::malloc(device, B_FP32);
    rlt::malloc(device, C_FP32);
    rlt::malloc(device, C_BF16_FP32_ACC);
    rlt::malloc(device, A_FP64);
    rlt::malloc(device, B_FP64);
    rlt::malloc(device, C_FP64);
    rlt::malloc(device, C_BF16_FP64);
    rlt::malloc(device, C_FP32_FP64);

    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);
    rlt::randn(device, A_FP64, rng);
    rlt::randn(device, B_FP64, rng);
    rlt::copy(device, device, A_FP64, A_FP32);
    rlt::copy(device, device, B_FP64, B_FP32);
    rlt::copy(device, device, A_FP64, A_BF16);
    rlt::copy(device, device, B_FP64, B_BF16);
    rlt::multiply(device, A_BF16, B_BF16, C_BF16);
    rlt::multiply(device, A_BF16, B_BF16, C_BF16_FP32_ACC);
    rlt::multiply(device, A_FP32, B_FP32, C_FP32);
    rlt::multiply(device, A_FP64, B_FP64, C_FP64);
    rlt::copy(device, device, C_BF16, C_BF16_FP64);
    rlt::copy(device, device, C_FP32, C_FP32_FP64);

    double diff_bf16_mul_bf16_acc = rlt::abs_diff(device, C_FP64, C_BF16_FP64);
    double diff_bf16_mul_fp32_acc = rlt::abs_diff(device, C_FP64, C_BF16_FP32_ACC);
    double diff_fp32_mul_fp32_acc = rlt::abs_diff(device, C_FP64, C_FP32);
    double diff_bf16_mul_bf16_acc_per_element = diff_bf16_mul_bf16_acc / (M * N);
    double diff_bf16_mul_fp32_acc_per_element = diff_bf16_mul_fp32_acc / (M * N);
    double diff_fp32_mul_fp32_acc_per_element = diff_fp32_mul_fp32_acc / (M * N);
    return {diff_bf16_mul_bf16_acc_per_element, diff_bf16_mul_fp32_acc_per_element, diff_fp32_mul_fp32_acc_per_element};
}

template <typename DEVICE, typename DEVICE::index_t N, typename DEVICE::index_t I = 0>
auto iterate(DEVICE& device) {
    using TI = typename DEVICE::index_t;
    constexpr TI UNIDIM = I == 0 ? 1 : 2 << I;
    std::vector<std::tuple<double, double, double>> results;
    results.push_back(test<DEVICE, UNIDIM, UNIDIM, UNIDIM>(device));
    if constexpr (I < N) {
        auto future_results = iterate<DEVICE, N, I + 1>(device);
        results.insert(results.end(), future_results.begin(), future_results.end());
    }
    return results;
}

TEST(TEST_CONTAINER_MIXED_PRECISION_BF16, MATMUL){
    using DEVICE = rlt::devices::DefaultCPU;
    DEVICE device;
    using TI = typename DEVICE::index_t;


    auto results = iterate<DEVICE, 10>(device);

    TI iteration = 0;
    for (auto [diff_bf16_mul_bf16_acc_per_element, diff_bf16_mul_fp32_acc_per_element, diff_fp32_mul_fp32_acc_per_element] : results) {
        std::cout << "Iteration: N M K: " << (iteration == 0 ? 1 : 2 << iteration) << std::endl;
        std::cout << "    bf16 mul, bf16 accumulation: diff: " << diff_bf16_mul_bf16_acc_per_element << std::endl;
        std::cout << "    bf16 mul, fp32 accumulation: diff: " << diff_bf16_mul_fp32_acc_per_element << std::endl;
        std::cout << "    fp32 mul, fp32 accumulation: diff: " << diff_fp32_mul_fp32_acc_per_element << std::endl;
        iteration++;
    }

}
