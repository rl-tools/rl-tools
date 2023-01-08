#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include <layer_in_c/operations/cpu.h>

#include "layer_in_c/nn_models/persist.h"
#include "layer_in_c/nn_models/operations_cpu.h"

#include "../utils/utils.h"

#include "default_network_mlp.h"
#include "../utils/nn_comparison_mlp.h"

#include <layer_in_c/utils/persist.h>

#include <random>
TEST(LAYER_IN_C_NN_PERSIST, Saving) {

    NetworkType network_1, network_2;
    std::mt19937 rng(2);
    lic::init_weights(network_1, rng);
    lic::init_weights(network_2, rng);
    {
        auto output_file = HighFive::File(std::string("test.hdf5"), HighFive::File::Overwrite);
        lic::save(network_1, output_file.createGroup("three_layer_fc"));
    }

    DTYPE diff_pre_load = abs_diff(network_1, network_2);
    ASSERT_GT(diff_pre_load, 10);
    std::cout << "diff_pre_load: " << diff_pre_load << std::endl;
    {
        auto input_file = HighFive::File(std::string("test.hdf5"), HighFive::File::ReadOnly);
        lic::load(network_2, input_file.getGroup("three_layer_fc"));
    }
    DTYPE diff_post_load = abs_diff(network_1, network_2);
    ASSERT_EQ(diff_post_load, 0);
    std::cout << "diff_post_load: " << diff_post_load << std::endl;
}

