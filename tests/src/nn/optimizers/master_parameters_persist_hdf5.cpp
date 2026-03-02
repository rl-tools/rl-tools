#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/persist.h>
#include <rl_tools/nn/optimizers/sgd/instance/persist.h>

#include <filesystem>
#include <string>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using TI = typename DEVICE::index_t;
using SHAPE = rlt::tensor::Shape<TI, 2, 3>;

using TYPE_POLICY_NO_MASTER = rlt::numeric_types::Policy<float>;
using TYPE_POLICY_MASTER_DIFF = rlt::numeric_types::Policy<float, rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, double>>;

template <typename TYPE_POLICY>
using SGD_PARAMETER_SPEC = rlt::nn::parameters::SGD::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true, false>;

template <typename TYPE_POLICY>
using ADAM_PARAMETER_SPEC = rlt::nn::parameters::Adam::Specification<TYPE_POLICY, TI, SHAPE, rlt::nn::parameters::groups::Normal, rlt::nn::parameters::categories::Weights, true, false>;

template <typename TYPE_POLICY>
using SGD_PARAMETER = rlt::nn::parameters::SGD::Instance<SGD_PARAMETER_SPEC<TYPE_POLICY>>;

template <typename TYPE_POLICY>
using ADAM_PARAMETER = rlt::nn::parameters::Adam::Instance<ADAM_PARAMETER_SPEC<TYPE_POLICY>>;

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS_PERSIST_HDF5, SGD_LOAD_FALLBACK_FROM_LEGACY_GROUP) {
    DEVICE device;
    SGD_PARAMETER<TYPE_POLICY_NO_MASTER> saved_no_master;
    SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF> loaded_with_master;
    rlt::malloc(device, saved_no_master);
    rlt::malloc(device, loaded_with_master);

    rlt::set_all(device, saved_no_master.parameters, 1.0f);
    rlt::set_all(device, saved_no_master.gradient, 2.0f);
    rlt::set_all(device, saved_no_master.velocity, 3.0f);
    rlt::set_all(device, loaded_with_master.parameters, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient, -1.0f);
    rlt::set_all(device, loaded_with_master.velocity, -1.0f);
    rlt::set_all(device, loaded_with_master.master_parameters, -9.0);

    const std::string path = "test_nn_optimizers_master_parameters_sgd_fallback.h5";
    {
        auto output_file = HighFive::File(path, HighFive::File::Overwrite);
        rlt::persist::backends::hdf5::Group<> group = {output_file.createGroup("sgd")};
        rlt::save(device, saved_no_master, group);
    }
    {
        auto input_file = HighFive::File(path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, input_file, "sgd");
        ASSERT_TRUE(rlt::load(device, loaded_with_master, group));
    }

    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.parameters, loaded_with_master.parameters), 0);
    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.gradient, loaded_with_master.gradient), 0);
    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.velocity, loaded_with_master.velocity), 0);
    EXPECT_EQ(rlt::abs_diff(device, loaded_with_master.parameters, loaded_with_master.master_parameters), 0);

    std::filesystem::remove(path);
    rlt::free(device, loaded_with_master);
    rlt::free(device, saved_no_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS_PERSIST_HDF5, SGD_ROUNDTRIP_WITH_MASTER_GROUP) {
    DEVICE device;
    SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF> saved_with_master;
    SGD_PARAMETER<TYPE_POLICY_MASTER_DIFF> loaded_with_master;
    rlt::malloc(device, saved_with_master);
    rlt::malloc(device, loaded_with_master);

    rlt::set_all(device, saved_with_master.parameters, 1.0f);
    rlt::set_all(device, saved_with_master.gradient, 2.0f);
    rlt::set_all(device, saved_with_master.velocity, 3.0f);
    rlt::set_all(device, saved_with_master.master_parameters, 4.0);
    rlt::set_all(device, loaded_with_master.parameters, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient, -1.0f);
    rlt::set_all(device, loaded_with_master.velocity, -1.0f);
    rlt::set_all(device, loaded_with_master.master_parameters, -1.0);

    const std::string path = "test_nn_optimizers_master_parameters_sgd_roundtrip.h5";
    {
        auto output_file = HighFive::File(path, HighFive::File::Overwrite);
        rlt::persist::backends::hdf5::Group<> group = {output_file.createGroup("sgd")};
        rlt::save(device, saved_with_master, group);
    }
    {
        auto input_file = HighFive::File(path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, input_file, "sgd");
        ASSERT_TRUE(rlt::load(device, loaded_with_master, group));
    }

    EXPECT_EQ(rlt::abs_diff(device, saved_with_master, loaded_with_master), 0);

    std::filesystem::remove(path);
    rlt::free(device, loaded_with_master);
    rlt::free(device, saved_with_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS_PERSIST_HDF5, ADAM_LOAD_FALLBACK_FROM_LEGACY_GROUP) {
    DEVICE device;
    ADAM_PARAMETER<TYPE_POLICY_NO_MASTER> saved_no_master;
    ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF> loaded_with_master;
    rlt::malloc(device, saved_no_master);
    rlt::malloc(device, loaded_with_master);

    rlt::set_all(device, saved_no_master.parameters, 1.0f);
    rlt::set_all(device, saved_no_master.gradient, 2.0f);
    rlt::set_all(device, saved_no_master.gradient_first_order_moment, 3.0f);
    rlt::set_all(device, saved_no_master.gradient_second_order_moment, 4.0f);
    rlt::set_all(device, loaded_with_master.parameters, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient_first_order_moment, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient_second_order_moment, -1.0f);
    rlt::set_all(device, loaded_with_master.master_parameters, -9.0);

    const std::string path = "test_nn_optimizers_master_parameters_adam_fallback.h5";
    {
        auto output_file = HighFive::File(path, HighFive::File::Overwrite);
        rlt::persist::backends::hdf5::Group<> group = {output_file.createGroup("adam")};
        rlt::save(device, saved_no_master, group);
    }
    {
        auto input_file = HighFive::File(path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, input_file, "adam");
        ASSERT_TRUE(rlt::load(device, loaded_with_master, group));
    }

    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.parameters, loaded_with_master.parameters), 0);
    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.gradient, loaded_with_master.gradient), 0);
    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.gradient_first_order_moment, loaded_with_master.gradient_first_order_moment), 0);
    EXPECT_EQ(rlt::abs_diff(device, saved_no_master.gradient_second_order_moment, loaded_with_master.gradient_second_order_moment), 0);
    EXPECT_EQ(rlt::abs_diff(device, loaded_with_master.parameters, loaded_with_master.master_parameters), 0);

    std::filesystem::remove(path);
    rlt::free(device, loaded_with_master);
    rlt::free(device, saved_no_master);
}

TEST(RL_TOOLS_NN_OPTIMIZERS_MASTER_PARAMETERS_PERSIST_HDF5, ADAM_ROUNDTRIP_WITH_MASTER_GROUP) {
    DEVICE device;
    ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF> saved_with_master;
    ADAM_PARAMETER<TYPE_POLICY_MASTER_DIFF> loaded_with_master;
    rlt::malloc(device, saved_with_master);
    rlt::malloc(device, loaded_with_master);

    rlt::set_all(device, saved_with_master.parameters, 1.0f);
    rlt::set_all(device, saved_with_master.gradient, 2.0f);
    rlt::set_all(device, saved_with_master.gradient_first_order_moment, 3.0f);
    rlt::set_all(device, saved_with_master.gradient_second_order_moment, 4.0f);
    rlt::set_all(device, saved_with_master.master_parameters, 5.0);
    rlt::set_all(device, loaded_with_master.parameters, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient_first_order_moment, -1.0f);
    rlt::set_all(device, loaded_with_master.gradient_second_order_moment, -1.0f);
    rlt::set_all(device, loaded_with_master.master_parameters, -1.0);

    const std::string path = "test_nn_optimizers_master_parameters_adam_roundtrip.h5";
    {
        auto output_file = HighFive::File(path, HighFive::File::Overwrite);
        rlt::persist::backends::hdf5::Group<> group = {output_file.createGroup("adam")};
        rlt::save(device, saved_with_master, group);
    }
    {
        auto input_file = HighFive::File(path, HighFive::File::ReadOnly);
        auto group = rlt::get_group(device, input_file, "adam");
        ASSERT_TRUE(rlt::load(device, loaded_with_master, group));
    }

    EXPECT_EQ(rlt::abs_diff(device, saved_with_master, loaded_with_master), 0);

    std::filesystem::remove(path);
    rlt::free(device, loaded_with_master);
    rlt::free(device, saved_with_master);
}
