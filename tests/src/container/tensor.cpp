#include <gtest/gtest.h>
#include <iostream>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers.h>
namespace rlt = rl_tools;

template <typename INPUT>
void test_shape_operations(int length){
    ASSERT_TRUE(length == rlt::length(INPUT{}));
    using APPEND = rlt::tensor::Append<INPUT, 5>;
    static_assert((rlt::length(INPUT{}) + 1) == rlt::length(APPEND{}));
    using PREPEND = rlt::tensor::Prepend<INPUT, 10>;
    static_assert((rlt::length(INPUT{}) + 1) == rlt::length(PREPEND{}));

    if constexpr(rlt::length(INPUT{}) > 1){
        using POP_FRONT = rlt::tensor::PopFront<INPUT>;
        static_assert(rlt::length(INPUT{}) == (rlt::length(POP_FRONT{}) + 1));
        using POP_BACK = rlt::tensor::PopBack<INPUT>;
        static_assert(rlt::length(INPUT{}) == (rlt::length(POP_BACK{}) + 1));

        if constexpr (rlt::length(INPUT{}) >= 3){
            ASSERT_TRUE(rlt::get<0>(POP_FRONT{}) == rlt::get<1>(INPUT{}));
            ASSERT_TRUE(rlt::get<1>(POP_FRONT{}) == rlt::get<2>(INPUT{}));
        }
        if constexpr(rlt::length(INPUT{}) >= 3){
            ASSERT_TRUE(rlt::get<0>(POP_BACK{}) == rlt::get<0>(INPUT{}));
            ASSERT_TRUE(rlt::get<1>(POP_BACK{}) == rlt::get<1>(INPUT{}));
        }
    }


    if constexpr(rlt::length(INPUT{}) >= 3){
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 0>;
            static_assert(rlt::get<0>(REPLACE{}) == 10);
            static_assert(rlt::get<1>(REPLACE{}) == rlt::get<1>(INPUT{}));
            static_assert(rlt::get<2>(REPLACE{}) == rlt::get<2>(INPUT{}));
        }
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 1>;
            static_assert(rlt::get<1>(REPLACE{}) == 10);
            static_assert(rlt::get<0>(REPLACE{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<2>(REPLACE{}) == rlt::get<2>(INPUT{}));
        }
        {
            using REPLACE = rlt::tensor::Replace<INPUT, 10, 2>;
            static_assert(rlt::get<2>(REPLACE{}) == 10);
            static_assert(rlt::get<0>(REPLACE{}) == rlt::get<0>(INPUT{}));
            static_assert(rlt::get<1>(REPLACE{}) == rlt::get<1>(INPUT{}));
        }
    }
    {

        using REPLACE = rlt::tensor::Replace<INPUT, 10, 0>;
        static_assert(rlt::get<0>(REPLACE{}) == 10);
    }
    {
        using REPLACE = rlt::tensor::Replace<INPUT, 1337, rlt::length(INPUT{})-1>;
        static_assert(rlt::get<rlt::length(INPUT{})-1>(REPLACE{}) == 1337);
    }

    if constexpr(rlt::length(INPUT{}) == 3){
        using PRODUCT = rlt::tensor::Product<INPUT>;
        ASSERT_TRUE(rlt::get<0>(PRODUCT{}) == rlt::get<0>(INPUT{}) * rlt::get<1>(INPUT{}) * rlt::get<2>(INPUT{}));
    }
}

TEST(RL_TOOLS_TENSOR_TEST, SHAPE_OPERATIONS){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3, 4>>(3);
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3>>(2);
    test_shape_operations<rlt::tensor::Shape<TI, 2>>(1);
    test_shape_operations<rlt::tensor::Shape<TI, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0>>(10);
}

TEST(RL_TOOLS_TENSOR_TEST, TENSOR){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    using SHAPE = rlt::tensor::Shape<TI, 2, 3, 4>;
    using STRIDE = rlt::tensor::RowMajorStride<SHAPE>;
    std::cout << "dim[0]: " << rlt::get<0>(SHAPE{}) << " stride[0]: " << rlt::get<0>(STRIDE{}) << std::endl;
    std::cout << "dim[1]: " << rlt::get<1>(SHAPE{}) << " stride[1]: " << rlt::get<1>(STRIDE{}) << std::endl;
    std::cout << "dim[2]: " << rlt::get<2>(SHAPE{}) << " stride[2]: " << rlt::get<2>(STRIDE{}) << std::endl;
}
