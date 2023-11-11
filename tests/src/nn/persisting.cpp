#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <rl_tools/operations/cpu.h>

#include "rl_tools/nn_models/persist.h"
#include "rl_tools/nn_models/operations_cpu.h"

#include "../utils/utils.h"

#include "default_network_mlp.h"
#include "../utils/nn_comparison_mlp.h"

#include <rl_tools/utils/persist.h>

#include <random>
TEST(RL_TOOLS_NN_PERSIST, Saving) {

    NN_DEVICE device;
    OPTIMIZER optimizer;
    NetworkType network_1, network_2;
    rlt::malloc(device, network_1);
    rlt::malloc(device, network_2);
    std::mt19937 rng(2);
    rlt::init_weights(device, network_1, rng);
    rlt::init_weights(device, network_2, rng);
    rlt::reset_forward_state(device, network_1);
    rlt::reset_optimizer_state(device, optimizer, network_1);
    rlt::zero_gradient(device, network_1);
    rlt::reset_forward_state(device, network_2);
    rlt::reset_optimizer_state(device, optimizer, network_2);
    rlt::zero_gradient(device, network_2);
    rlt::increment(network_1.input_layer.weights.gradient_first_order_moment, 2, 3, 10);
    {
        auto output_file = HighFive::File(std::string("test.hdf5"), HighFive::File::Overwrite);
        rlt::save(device, network_1, output_file.createGroup("three_layer_fc"));
    }

    DTYPE diff_pre_load = abs_diff(device, network_1, network_2);
    ASSERT_GT(diff_pre_load, 10);
    std::cout << "diff_pre_load: " << diff_pre_load << std::endl;
    {
        auto input_file = HighFive::File(std::string("test.hdf5"), HighFive::File::ReadOnly);
        rlt::load(device, network_2, input_file.getGroup("three_layer_fc"));
    }
    rlt::reset_forward_state(device, network_1);
    rlt::reset_forward_state(device, network_2);
    DTYPE diff_post_load = abs_diff(device, network_1, network_2);
    ASSERT_EQ(diff_post_load, 0);
    std::cout << "diff_post_load: " << diff_post_load << std::endl;
}

