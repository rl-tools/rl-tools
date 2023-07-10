#include <backprop_tools/operations/cpu.h>


#include <gtest/gtest.h>

namespace bpt = backprop_tools;


TEST(BACKPROP_TOOLS_TEST_CONTAINER, SLICE){
    using DEVICE = bpt::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    bpt::MatrixDynamic<bpt::matrix::Specification<float, typename DEVICE::index_t, 3, 3>> m;
    bpt::malloc(device, m);
    bpt::set(m, 0, 0, 1);
    bpt::set(m, 0, 1, 2);
    bpt::set(m, 0, 2, 3);
    bpt::set(m, 1, 0, 4);
    bpt::set(m, 1, 1, 5);
    bpt::set(m, 1, 2, 6);
    bpt::set(m, 2, 0, 7);
    bpt::set(m, 2, 1, 8);
    bpt::set(m, 2, 2, 9);
    bpt::print(device, m);
    auto m2 = bpt::view<DEVICE, decltype(m)::SPEC, 2, 2>(device, m, 0, 1);
    bpt::print(device, m2);
    ASSERT_FLOAT_EQ(bpt::get(m2, 0, 0), 2);
    ASSERT_FLOAT_EQ(bpt::get(m2, 0, 1), 3);
    ASSERT_FLOAT_EQ(bpt::get(m2, 1, 0), 5);
    ASSERT_FLOAT_EQ(bpt::get(m2, 1, 1), 6);

    std::cout << "transpose: " << std::endl;
    auto m3 = bpt::view_transpose(device, m);
    bpt::print(device, m3);
    bpt::free(device, m);

    bpt::MatrixDynamic<bpt::matrix::Specification<float, typename DEVICE::index_t, 17, 15>> m4;
    bpt::malloc(device, m4);
    //init with random data
    for(typename DEVICE::index_t row_i = 0; row_i < decltype(m4)::ROWS; row_i++){
        for(typename DEVICE::index_t col_i = 0; col_i < decltype(m4)::COLS; col_i++){
            bpt::set(m4, row_i, col_i, row_i * col_i);
        }
    }
    auto m5 = bpt::view<DEVICE, decltype(m4)::SPEC, 5, 5>(device, m4, 3, 4);
    bpt::print(device, m5);
    for(typename DEVICE::index_t row_i = 0; row_i < decltype(m5)::ROWS; row_i++){
        for(typename DEVICE::index_t col_i = 0; col_i < decltype(m5)::COLS; col_i++){
            ASSERT_FLOAT_EQ(bpt::get(m5, row_i, col_i), (row_i + 3) * (col_i + 4));
        }
    }
    bpt::free(device, m4);

}
template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH, int VIEW_ROWS, int VIEW_COLS>
void test_view(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> m_dense;
    bpt::malloc(device, m);
    bpt::malloc(device, m_dense);
    bpt::randn(device, m, rng);
    bpt::copy(device, device, m_dense, m);

    for(TI row_i = 0; row_i < ROWS-VIEW_ROWS; row_i++){
        for(TI col_i = 0; col_i < COLS-VIEW_COLS; col_i++){
            auto view = bpt::view(device, m, bpt::matrix::ViewSpec<VIEW_ROWS, VIEW_COLS>(), row_i, col_i);
            auto view_dense = bpt::view(device, m_dense, bpt::matrix::ViewSpec<VIEW_ROWS, VIEW_COLS>(), row_i, col_i);
            auto abs_diff = bpt::abs_diff(device, view, view_dense);
            ASSERT_FLOAT_EQ(abs_diff, 0);
        }
    }

    bpt::free(device, m);
    bpt::free(device, m_dense);

}
TEST(BACKPROP_TOOLS_TEST_CONTAINER, VIEW) {
    test_view<10, 10, 10, 1, 5, 5>();
    test_view<10, 10, 20, 2, 5, 5>();
    test_view<15, 13, 100, 3, 3, 2>();
    test_view<15, 13, 3, 100, 1, 1>();
    test_view<15, 13, 3, 100, 10, 1>();
    test_view<15, 13, 3, 100, 1, 10>();
}

template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_view_col(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::RowMajorAlignment<TI, 1>>> m_dense;
    bpt::malloc(device, m);
    bpt::malloc(device, m_dense);
    bpt::randn(device, m, rng);
    bpt::copy(device, device, m_dense, m);

    for(TI row_i = 0; row_i < ROWS; row_i++){
        auto row_m = bpt::row(device, m, row_i);
        auto row_m_dense = bpt::row(device, m_dense, row_i);
        auto abs_diff = bpt::abs_diff(device, row_m, row_m_dense);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }

    for(TI col_i = 0; col_i < COLS; col_i++){
        auto col_m = bpt::col(device, m, col_i);
        auto col_m_dense = bpt::col(device, m_dense, col_i);
        auto abs_diff = bpt::abs_diff(device, col_m, col_m_dense);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }
    bpt::free(device, m);
    bpt::free(device, m_dense);

}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, VIEW_ROW_COL) {
    test_view_col<10, 10, 10, 1>();
    test_view_col<10, 10, 20, 2>();
    test_view_col<15, 13, 100, 3>();
    test_view_col<15, 13, 3, 100>();
}

template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_is_nan(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    bpt::malloc(device, m);
    bpt::randn(device, m, rng);

    bool is_nan_m = bpt::is_nan(device, m);
    ASSERT_TRUE(!is_nan_m);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::quiet_NaN());
    is_nan_m = bpt::is_nan(device, m);
    ASSERT_TRUE(is_nan_m);

    bpt::free(device, m);
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, IS_NAN) {
    test_is_nan<10, 10, 10, 1>();
    test_is_nan<10, 10, 20, 2>();
    test_is_nan<15, 13, 100, 3>();
    test_is_nan<15, 13, 3, 100>();
}
template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_is_finite(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS, bpt::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    bpt::malloc(device, m);
    bpt::randn(device, m, rng);

    bool is_finite = bpt::is_finite(device, m);
    ASSERT_TRUE(is_finite);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::infinity());
    bpt::print(device, m);
    is_finite = bpt::is_finite(device, m);
    ASSERT_TRUE(!is_finite);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::quiet_NaN());
    bpt::print(device, m);
    is_finite = bpt::is_finite(device, m);
    ASSERT_TRUE(!is_finite);

    bpt::free(device, m);
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, IS_FINITE) {
    test_is_finite<10, 10, 10, 1>();
    test_is_finite<10, 10, 20, 2>();
    test_is_finite<15, 13, 100, 3>();
    test_is_finite<15, 13, 3, 100>();
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, WRAP) {
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    T test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto m = bpt::wrap<DEVICE, T, DIM>(device, test);
    bpt::print(device, m);
    for(TI i = 0; i < DIM; i++){
        ASSERT_FLOAT_EQ(get(m, 0, i), test[i]);
    }
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, MIN_DETERMINISTIC) {
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    T test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto m = bpt::wrap<DEVICE, T, DIM>(device, test);
    bpt::print(device, m);
    T min = bpt::min(device, m);
    ASSERT_FLOAT_EQ(min, 1);
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, MAX_DETERMINISTIC) {
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    T test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto m = bpt::wrap<DEVICE, T, DIM>(device, test);
    bpt::print(device, m);
    T max = bpt::max(device, m);
    ASSERT_FLOAT_EQ(max, 11);
}

template <int ROWS, int COLS>
void test_max_stochastic(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS>> m;
    bpt::malloc(device, m);
    for(TI test_i = 0; test_i < 10; test_i++){
        bpt::randn(device, m, rng);
        T max = bpt::max(device, m);
        for(TI row_i = 0; row_i < ROWS; row_i++){
            for(TI col_i = 0; col_i < COLS; col_i++){
                ASSERT_TRUE(get(m, row_i, col_i) <= max);
            }
        }
    }
}
TEST(BACKPROP_TOOLS_TEST_CONTAINER, MAX_STOCHASTIC) {
    test_max_stochastic<10, 10>();
    test_max_stochastic<10, 1000>();
    test_max_stochastic<1, 1>();
    test_max_stochastic<1, 10>();
    test_max_stochastic<10, 1>();
}

TEST(BACKPROP_TOOLS_TEST_CONTAINER, ARGMAX_DETERMINISTIC) {
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    {
        T test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
        auto m = bpt::wrap<DEVICE, T, DIM>(device, test);
        bpt::print(device, m);
        TI am = bpt::argmax_row(device, m);
        ASSERT_FLOAT_EQ(am, 10);
    }
    {
        T test[] = {1, 2, 3, 4, 50, 6, 7, 8, 9, 10, 11};
        auto m = bpt::wrap<DEVICE, T, DIM>(device, test);
        bpt::print(device, m);
        TI am = bpt::argmax_row(device, m);
        ASSERT_FLOAT_EQ(am, 4);
    }
}

template <int ROWS, int COLS>
void test_argmax_stochastic(){
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM());
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, COLS>> m;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, ROWS, 1>> am;
    bpt::malloc(device, m);
    bpt::malloc(device, am);
    for(TI test_i = 0; test_i < 10; test_i++){
        bpt::randn(device, m, rng);
        bpt::argmax_row_wise(device, m, am);
        for(TI row_i = 0; row_i < ROWS; row_i++){
            T row_max = get(m, row_i, get(am, row_i, 0));
            for(TI col_i = 0; col_i < COLS; col_i++){
                if(!(get(m, row_i, col_i) <= row_max)){
                    bpt::print(device, m);
                }
                ASSERT_TRUE(get(m, row_i, col_i) <= row_max);
            }
        }
    }
}
TEST(BACKPROP_TOOLS_TEST_CONTAINER, ARGMAX_STOCHASTIC) {
    test_argmax_stochastic<10, 10>();
    test_argmax_stochastic<10, 1000>();
    test_argmax_stochastic<1, 1>();
    test_argmax_stochastic<1, 10>();
    test_argmax_stochastic<10, 1>();
}


TEST(BACKPROP_TOOLS_TEST_CONTAINER, MATRIX_MULTIPLICATION_GENERIC) {
    using DEVICE = bpt::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 2, 2>> A, B, C, C_target;
    bpt::malloc(device, A);
    bpt::malloc(device, B);
    bpt::malloc(device, C);
    bpt::malloc(device, C_target);
    set(A, 0, 0, -0.259093);
    set(A, 0, 1, -1.498961);
    set(A, 1, 0, +0.119264);
    set(A, 1, 1, +0.458181);

    set(B, 0, 0, +0.394975);
    set(B, 0, 1, +0.044197);
    set(B, 1, 0, -0.636256);
    set(B, 1, 1, +1.731264);

    set(C_target, 0, 0, -0.259093 * +0.394975 + -1.498961 * -0.636256);
    set(C_target, 0, 1, -0.259093 * +0.044197 + -1.498961 * +1.731264);
    set(C_target, 1, 0, +0.119264 * +0.394975 + +0.458181 * -0.636256);
    set(C_target, 1, 1, +0.119264 * +0.044197 + +0.458181 * +1.731264);
    bpt::print(device, C_target);

    bpt::multiply(device, A, B, C);
    bpt::print(device, C);
    auto diff = bpt::abs_diff(device, C_target, C);
    std::cout << "Matrix mul diff: " << diff << std::endl;
    ASSERT_TRUE(diff < 1e-6);
}

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#define BACKPROP_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#include <backprop_tools/operations/cpu_mkl.h>
TEST(BACKPROP_TOOLS_TEST_CONTAINER, MATRIX_MULTIPLICATION_MKL) {
    using DEVICE = bpt::devices::DefaultCPU_MKL;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 2, 2>> A, B, C, C_target;
    bpt::malloc(device, A);
    bpt::malloc(device, B);
    bpt::malloc(device, C);
    bpt::malloc(device, C_target);
    set(A, 0, 0, -0.259093);
    set(A, 0, 1, -1.498961);
    set(A, 1, 0, +0.119264);
    set(A, 1, 1, +0.458181);

    set(B, 0, 0, +0.394975);
    set(B, 0, 1, +0.044197);
    set(B, 1, 0, -0.636256);
    set(B, 1, 1, +1.731264);

    set(C_target, 0, 0, -0.259093 * +0.394975 + -1.498961 * -0.636256);
    set(C_target, 0, 1, -0.259093 * +0.044197 + -1.498961 * +1.731264);
    set(C_target, 1, 0, +0.119264 * +0.394975 + +0.458181 * -0.636256);
    set(C_target, 1, 1, +0.119264 * +0.044197 + +0.458181 * +1.731264);
    bpt::print(device, C_target);

    bpt::multiply(device, A, B, C);
    bpt::print(device, C);
    auto diff = bpt::abs_diff(device, C_target, C);
    std::cout << "Matrix mul diff: " << diff << std::endl;
    ASSERT_TRUE(diff < 1e-6);
}
#endif
