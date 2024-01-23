#include <gtest/gtest.h>
#include <iostream>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers.h>
namespace rlt = rl_tools;

template <typename INPUT>
void test(int length){
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

    if constexpr(rlt::length(INPUT{}) == 3){
        using PRODUCT = rlt::tensor::Product<INPUT>;
        ASSERT_TRUE(rlt::get<0>(PRODUCT{}) == rlt::get<0>(INPUT{}) * rlt::get<1>(INPUT{}) * rlt::get<2>(INPUT{}));
    }
}

TEST(RL_TOOLS_TENSOR, TEST){
    using DEVICE = rlt::devices::DefaultCPU;
    using T = double;
    using TI = typename DEVICE::index_t;
    test<rlt::tensor::Shape<TI, 2, 3, 4>>(3);
    test<rlt::tensor::Shape<TI, 2, 3>>(2);
    test<rlt::tensor::Shape<TI, 2>>(1);
    test<rlt::tensor::Shape<TI, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0>>(10);
}
