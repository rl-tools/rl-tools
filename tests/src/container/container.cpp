#include <layer_in_c/operations/cpu.h>


#include <gtest/gtest.h>

namespace lic = layer_in_c;


TEST(LAYER_IN_C_TEST_CONTAINER, SLICE){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    lic::MatrixDynamic<lic::matrix::Specification<float, typename DEVICE::index_t, 3, 3>> m;
    lic::malloc(device, m);
    lic::set(m, 0, 0, 1);
    lic::set(m, 0, 1, 2);
    lic::set(m, 0, 2, 3);
    lic::set(m, 1, 0, 4);
    lic::set(m, 1, 1, 5);
    lic::set(m, 1, 2, 6);
    lic::set(m, 2, 0, 7);
    lic::set(m, 2, 1, 8);
    lic::set(m, 2, 2, 9);
    lic::print(device, m);
    auto m2 = lic::view<DEVICE, decltype(m)::SPEC, 2, 2>(device, m, 0, 1);
    lic::print(device, m2);
    ASSERT_FLOAT_EQ(lic::get(m2, 0, 0), 2);
    ASSERT_FLOAT_EQ(lic::get(m2, 0, 1), 3);
    ASSERT_FLOAT_EQ(lic::get(m2, 1, 0), 5);
    ASSERT_FLOAT_EQ(lic::get(m2, 1, 1), 6);

    std::cout << "transpose: " << std::endl;
    auto m3 = lic::view_transpose(device, m);
    lic::print(device, m3);
    lic::free(device, m);

    lic::MatrixDynamic<lic::matrix::Specification<float, typename DEVICE::index_t, 17, 15>> m4;
    lic::malloc(device, m4);
    //init with random data
    for(typename DEVICE::index_t row_i = 0; row_i < decltype(m4)::ROWS; row_i++){
        for(typename DEVICE::index_t col_i = 0; col_i < decltype(m4)::COLS; col_i++){
            lic::set(m4, row_i, col_i, row_i * col_i);
        }
    }
    auto m5 = lic::view<DEVICE, decltype(m4)::SPEC, 5, 5>(device, m4, 3, 4);
    lic::print(device, m5);
    for(typename DEVICE::index_t row_i = 0; row_i < decltype(m5)::ROWS; row_i++){
        for(typename DEVICE::index_t col_i = 0; col_i < decltype(m5)::COLS; col_i++){
            ASSERT_FLOAT_EQ(lic::get(m5, row_i, col_i), (row_i + 3) * (col_i + 4));
        }
    }
    lic::free(device, m4);

}
template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH, int VIEW_ROWS, int VIEW_COLS>
void test_view(){
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> m_dense;
    lic::malloc(device, m);
    lic::malloc(device, m_dense);
    lic::randn(device, m, rng);
    lic::copy(device, device, m_dense, m);

    for(TI row_i = 0; row_i < ROWS-VIEW_ROWS; row_i++){
        for(TI col_i = 0; col_i < COLS-VIEW_COLS; col_i++){
            auto view = lic::view(device, m, lic::matrix::ViewSpec<VIEW_ROWS, VIEW_COLS>(), row_i, col_i);
            auto view_dense = lic::view(device, m_dense, lic::matrix::ViewSpec<VIEW_ROWS, VIEW_COLS>(), row_i, col_i);
            auto abs_diff = lic::abs_diff(device, view, view_dense);
            ASSERT_FLOAT_EQ(abs_diff, 0);
        }
    }

    lic::free(device, m);
    lic::free(device, m_dense);

}
TEST(LAYER_IN_C_TEST_CONTAINER, VIEW) {
    test_view<10, 10, 10, 1, 5, 5>();
    test_view<10, 10, 20, 2, 5, 5>();
    test_view<15, 13, 100, 3, 3, 2>();
    test_view<15, 13, 3, 100, 1, 1>();
    test_view<15, 13, 3, 100, 10, 1>();
    test_view<15, 13, 3, 100, 1, 10>();
}

template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_view_col(){
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::RowMajorAlignment<TI, 1>>> m_dense;
    lic::malloc(device, m);
    lic::malloc(device, m_dense);
    lic::randn(device, m, rng);
    lic::copy(device, device, m_dense, m);

    for(TI row_i = 0; row_i < ROWS; row_i++){
        auto row_m = lic::row(device, m, row_i);
        auto row_m_dense = lic::row(device, m_dense, row_i);
        auto abs_diff = lic::abs_diff(device, row_m, row_m_dense);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }

    for(TI col_i = 0; col_i < COLS; col_i++){
        auto col_m = lic::col(device, m, col_i);
        auto col_m_dense = lic::col(device, m_dense, col_i);
        auto abs_diff = lic::abs_diff(device, col_m, col_m_dense);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }
    lic::free(device, m);
    lic::free(device, m_dense);

}

TEST(LAYER_IN_C_TEST_CONTAINER, VIEW_ROW_COL) {
    test_view_col<10, 10, 10, 1>();
    test_view_col<10, 10, 20, 2>();
    test_view_col<15, 13, 100, 3>();
    test_view_col<15, 13, 3, 100>();
}

template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_is_nan(){
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    lic::malloc(device, m);
    lic::randn(device, m, rng);

    bool is_nan_m = lic::is_nan(device, m);
    ASSERT_TRUE(!is_nan_m);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::quiet_NaN());
    is_nan_m = lic::is_nan(device, m);
    ASSERT_TRUE(is_nan_m);

    lic::free(device, m);
}

TEST(LAYER_IN_C_TEST_CONTAINER, IS_NAN) {
    test_is_nan<10, 10, 10, 1>();
    test_is_nan<10, 10, 20, 2>();
    test_is_nan<15, 13, 100, 3>();
    test_is_nan<15, 13, 3, 100>();
}
template <int ROWS, int COLS, int ROW_PITCH, int COL_PITCH>
void test_is_finite(){
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, ROWS, COLS, lic::matrix::layouts::Fixed<TI, ROW_PITCH, COL_PITCH>>> m;
    lic::malloc(device, m);
    lic::randn(device, m, rng);

    bool is_finite = lic::is_finite(device, m);
    ASSERT_TRUE(is_finite);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::infinity());
    lic::print(device, m);
    is_finite = lic::is_finite(device, m);
    ASSERT_TRUE(!is_finite);

    set(m, ROWS/2, COLS/2, std::numeric_limits<T>::quiet_NaN());
    lic::print(device, m);
    is_finite = lic::is_finite(device, m);
    ASSERT_TRUE(!is_finite);

    lic::free(device, m);
}

TEST(LAYER_IN_C_TEST_CONTAINER, IS_FINITE) {
    test_is_finite<10, 10, 10, 1>();
    test_is_finite<10, 10, 20, 2>();
    test_is_finite<15, 13, 100, 3>();
    test_is_finite<15, 13, 3, 100>();
}

TEST(LAYER_IN_C_TEST_CONTAINER, WRAP) {
    using DEVICE = lic::devices::DefaultCPU;
    using T = float;
    using TI = DEVICE::index_t;
    constexpr int DIM = 11;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    T test[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    auto m = lic::wrap<DEVICE, T, DIM>(device, test);
    lic::print(device, m);
    for(TI i = 0; i < DIM; i++){
        ASSERT_FLOAT_EQ(get(m, 0, i), test[i]);
    }
}
