#include <gtest/gtest.h>
#include <iostream>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers.h>
namespace rlt = rl_tools;

TEST(RL_TOOLS_TENSOR, TEST){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using SHAPE2 = rlt::tensor::Append<SHAPE, 5>;
    using SHAPE3 = rlt::tensor::Prepend<SHAPE, 10>;
//    using STRIDE = rlt::tensor::Stride<TI, 12, 4, 1>;
//    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    using STRIDE = rlt::tensor::Product<SHAPE>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, STRIDE>> tensor;
    std::cout << "length SHAPE: " << rlt::length(SHAPE{}) << " SHAPE2: " << rlt::length(SHAPE2{}) << " SHAPE3: " << rlt::length(SHAPE3{}) << std::endl;
    std::cout << rlt::get<0>(SHAPE{}) << " : " << rlt::get<0>(SHAPE2{}) << " : " << rlt::get<0>(SHAPE3{}) << std::endl;
    std::cout << rlt::get<1>(SHAPE{}) << " : " << rlt::get<1>(SHAPE2{}) << " : " << rlt::get<1>(SHAPE3{}) << std::endl;
    std::cout << rlt::get<2>(SHAPE{}) << " : " << rlt::get<2>(SHAPE2{}) << " : " << rlt::get<2>(SHAPE3{}) << std::endl;
    std::cout <<                         " : " << rlt::get<3>(SHAPE2{}) << " : " << rlt::get<3>(SHAPE3{}) << std::endl;
    std::cout << rlt::product(SHAPE{}) << std::endl;
    std::cout << rlt::product(rlt::tensor::Shape<TI, 2, 3>{}) << std::endl;
    std::cout << rlt::get<0>(STRIDE{}) << std::endl;
    std::cout << rlt::get<1>(STRIDE{}) << std::endl;
    std::cout << rlt::get<2>(STRIDE{}) << std::endl;
//    std::cout << rlt::get<4>(SHAPE{}) << std::endl;
//    std::cout << rlt::get<5>(SHAPE{}) << std::endl;
}
