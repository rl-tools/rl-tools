#include <layer_in_c/operations/cpu.h>
#include <layer_in_c/containers/persist_code.h>

namespace lic = layer_in_c;


#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <filesystem>


TEST(LAYER_IN_C_CONTAINER_PERSIST_GENERIC, TEST){
    using DEVICE = lic::devices::DefaultCPU;
    using DTYPE = float;
    DEVICE device;
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    lic::Matrix<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, 3, 3>> m;
    lic::malloc(device, m);
    lic::randn(device, m, rng);
    lic::print(device, m);
    auto output = lic::save(device, m, "matrix_1");
    std::cout << "output: " << output << std::endl;
    std::filesystem::create_directories("data");
    std::ofstream file;
    file.open ("data/test_layer_in_c_container_persist_matrix.h");
    file << output;
    file.close();

    ASSERT_TRUE(true);
}