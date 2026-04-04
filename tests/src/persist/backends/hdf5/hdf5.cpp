#include <gtest/gtest.h>
#include <iostream>
#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>

namespace rlt = rl_tools;

using DEVICE = rl_tools::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, test) {
    DEVICE device;
    RNG rng;
    constexpr TI seed = 0;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 4, 3>>> A;
    rlt::malloc(device, A);
    rlt::randn(device, A, rng);
    rlt::print(device, A);
    std::string data_file_name = "test_persist_backends_hdf5.h5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;
    auto output_file = rl_tools::persist::backends::hdf5::File(data_file_path, rl_tools::persist::backends::hdf5::Mode::WRITE);
    auto group = rlt::create_group(device, output_file, "test");
    rlt::save(device, A, group, "A");


    using CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using SPEC = rlt::nn::layers::dense::Specification<CONFIG, CAPABILITY, rlt::tensor::Shape<TI, 1, 10, 15>>;
    rlt::nn::layers::dense::LayerForward<SPEC> layer;
    rlt::malloc(device, layer);
    rlt::init_weights(device, layer, rng);
    rlt::print(device, layer.weights.parameters);
    auto layer_group = rlt::create_group(device, group, "layer");
    rlt::save(device, layer, layer_group);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, tensor_1d) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);

    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string path = std::string(dp) + "/test_persist_hdf5_1d.h5";

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 17>>> original, loaded;
    rlt::malloc(device, original);
    rlt::malloc(device, loaded);
    rlt::randn(device, original, rng);

    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto g = rlt::create_group(device, file, "data");
        rlt::save(device, original, g, "tensor");
    }
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::READ);
        auto g = rlt::get_group(device, file, "data");
        ASSERT_TRUE(rlt::load(device, loaded, g, "tensor"));
    }
    ASSERT_EQ(rlt::abs_diff(device, original, loaded), 0);
    rlt::free(device, original);
    rlt::free(device, loaded);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, tensor_3d) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 2);

    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string path = std::string(dp) + "/test_persist_hdf5_3d.h5";

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 7, 5>>> original, loaded;
    rlt::malloc(device, original);
    rlt::malloc(device, loaded);
    rlt::randn(device, original, rng);

    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto g = rlt::create_group(device, file, "data");
        rlt::save(device, original, g, "tensor");
    }
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::READ);
        auto g = rlt::get_group(device, file, "data");
        ASSERT_TRUE(rlt::load(device, loaded, g, "tensor"));
    }
    ASSERT_EQ(rlt::abs_diff(device, original, loaded), 0);
    rlt::free(device, original);
    rlt::free(device, loaded);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, tensor_double) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 3);

    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string path = std::string(dp) + "/test_persist_hdf5_double.h5";

    rlt::Tensor<rlt::tensor::Specification<double, TI, rlt::tensor::Shape<TI, 4, 6>>> original, loaded;
    rlt::malloc(device, original);
    rlt::malloc(device, loaded);
    rlt::randn(device, original, rng);

    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto g = rlt::create_group(device, file, "data");
        rlt::save(device, original, g, "tensor");
    }
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::READ);
        auto g = rlt::get_group(device, file, "data");
        ASSERT_TRUE(rlt::load(device, loaded, g, "tensor"));
    }
    ASSERT_EQ(rlt::abs_diff(device, original, loaded), 0);
    rlt::free(device, original);
    rlt::free(device, loaded);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, non_contiguous_tensor) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 4);

    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string path = std::string(dp) + "/test_persist_hdf5_non_contiguous.h5";

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 5, 8, 3>>> full;
    rlt::malloc(device, full);
    rlt::randn(device, full, rng);

    auto nc_view = rlt::view<1>(device, full, 2);
    using NC_SPEC = decltype(nc_view)::SPEC;
    static_assert(!rlt::tensor::dense_row_major_layout<NC_SPEC>(), "view<1> should produce non-contiguous tensor");

    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto g = rlt::create_group(device, file, "data");
        rlt::save(device, nc_view, g, "nc");
    }

    rlt::Tensor<rlt::tensor::Specification<T, TI, typename NC_SPEC::SHAPE>> loaded;
    rlt::malloc(device, loaded);
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::READ);
        auto g = rlt::get_group(device, file, "data");
        ASSERT_TRUE(rlt::load(device, loaded, g, "nc"));
    }
    ASSERT_EQ(rlt::abs_diff(device, nc_view, loaded), 0);
    rlt::free(device, full);
    rlt::free(device, loaded);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, dense_layer_round_trip) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 5);

    const char *dp = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string path = std::string(dp) + "/test_persist_hdf5_dense.h5";

    using CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 8, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using CAPABILITY = rlt::nn::capability::Forward<>;
    using LAYER_SPEC = rlt::nn::layers::dense::Specification<CONFIG, CAPABILITY, rlt::tensor::Shape<TI, 1, 1, 12>>;
    rlt::nn::layers::dense::LayerForward<LAYER_SPEC> original, loaded;
    rlt::malloc(device, original);
    rlt::malloc(device, loaded);
    rlt::init_weights(device, original, rng);

    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto g = rlt::create_group(device, file, "layer");
        rlt::save(device, original, g);
    }
    {
        auto file = rlt::persist::backends::hdf5::File(path, rlt::persist::backends::hdf5::Mode::READ);
        auto g = rlt::get_group(device, file, "layer");
        ASSERT_TRUE(rlt::load(device, loaded, g));
    }

    T w_diff = rlt::abs_diff(device, original.weights.parameters, loaded.weights.parameters);
    T b_diff = rlt::abs_diff(device, original.biases.parameters, loaded.biases.parameters);
    ASSERT_EQ(w_diff, 0) << "Weight diff: " << w_diff;
    ASSERT_EQ(b_diff, 0) << "Bias diff: " << b_diff;
    rlt::free(device, original);
    rlt::free(device, loaded);
}

TEST(TEST_PERSIST_BACKENDS_HDF5_HDF5, matrix_round_trip) {
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    std::string data_file_name = "test_persist_backends_hdf5_matrix.h5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string data_file_path = std::string(data_path_stub) + "/" + data_file_name;

    using MATRIX_SPEC = rlt::matrix::Specification<T, TI, 7, 5>;
    rlt::Matrix<MATRIX_SPEC> original, loaded;
    rlt::malloc(device, original);
    rlt::malloc(device, loaded);
    rlt::randn(device, original, rng);

    {
        auto file = rlt::persist::backends::hdf5::File(data_file_path, rlt::persist::backends::hdf5::Mode::WRITE);
        auto group = rlt::create_group(device, file, "test");
        rlt::save(device, original, group, "matrix");
    }
    {
        auto file = rlt::persist::backends::hdf5::File(data_file_path, rlt::persist::backends::hdf5::Mode::READ);
        auto group = rlt::get_group(device, file, "test");
        ASSERT_TRUE(rlt::load(device, loaded, group, "matrix"));
    }

    T max_diff = rlt::abs_diff(device, original, loaded);
    ASSERT_EQ(max_diff, 0) << "Matrix round-trip should be lossless, got diff: " << max_diff;

    rlt::free(device, original);
    rlt::free(device, loaded);
}