#include <layer_in_c/operations/cpu.h>


#include <gtest/gtest.h>

namespace lic = layer_in_c;


TEST(LAYER_IN_C_TEST_CONTAINER, SLICE) {
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE::SPEC::LOGGING logger;
    DEVICE device(logger);
    lic::Matrix<lic::matrix::Specification<float, typename DEVICE::index_t, 3, 3>> m;
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

    auto m3 = lic::transpose(device, m);
    lic::print(device, m3);
    lic::free(device, m3);

    lic::Matrix<lic::matrix::Specification<float, typename DEVICE::index_t, 17, 15>> m4;
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
