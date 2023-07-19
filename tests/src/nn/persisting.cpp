#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <backprop_tools/operations/cpu.h>

#include "backprop_tools/nn_models/persist.h"
#include "backprop_tools/nn_models/operations_cpu.h"

#include "../utils/utils.h"

#include "default_network_mlp.h"
#include "../utils/nn_comparison_mlp.h"

#include <backprop_tools/utils/persist.h>

#include <random>
TEST(BACKPROP_TOOLS_NN_PERSIST, Saving) {

    NN_DEVICE::SPEC::LOGGING logger;
    NN_DEVICE device;
    OPTIMIZER optimizer;
    device.logger = &logger;
    NetworkType network_1, network_2;
    bpt::malloc(device, network_1);
    bpt::malloc(device, network_2);
    std::mt19937 rng(2);
    bpt::init_weights(device, network_1, rng);
    bpt::init_weights(device, network_2, rng);
    bpt::reset_forward_state(device, network_1);
    bpt::reset_optimizer_state(device, optimizer, network_1);
    bpt::zero_gradient(device, network_1);
    bpt::reset_forward_state(device, network_2);
    bpt::reset_optimizer_state(device, optimizer, network_2);
    bpt::zero_gradient(device, network_2);
    bpt::increment(network_1.input_layer.weights.gradient_first_order_moment, 2, 3, 10);
    {
        auto output_file = HighFive::File(std::string("test.hdf5"), HighFive::File::Overwrite);
        bpt::save(device, network_1, output_file.createGroup("three_layer_fc"));
    }

    DTYPE diff_pre_load = abs_diff(device, network_1, network_2);
    ASSERT_GT(diff_pre_load, 10);
    std::cout << "diff_pre_load: " << diff_pre_load << std::endl;
    {
        auto input_file = HighFive::File(std::string("test.hdf5"), HighFive::File::ReadOnly);
        bpt::load(device, network_2, input_file.getGroup("three_layer_fc"));
    }
    bpt::reset_forward_state(device, network_1);
    bpt::reset_forward_state(device, network_2);
    DTYPE diff_post_load = abs_diff(device, network_1, network_2);
    ASSERT_EQ(diff_post_load, 0);
    std::cout << "diff_post_load: " << diff_post_load << std::endl;
}

