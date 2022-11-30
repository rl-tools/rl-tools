#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

#include "layer_in_c/nn_models/models.h"
#include "layer_in_c/nn_models/persist.h"
#include <layer_in_c/nn/nn.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/utils/rng_std.h>

#include "../utils/utils.h"

namespace lic = layer_in_c;

#define DTYPE float
constexpr size_t INPUT_DIM = 17;
constexpr size_t LAYER_1_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_1_FN =  lic::nn::activation_functions::RELU;
constexpr size_t LAYER_2_DIM = 50;
constexpr lic::nn::activation_functions::ActivationFunction LAYER_2_FN = lic::nn::activation_functions::RELU;
constexpr size_t OUTPUT_DIM = 13;
constexpr lic::nn::activation_functions::ActivationFunction OUTPUT_FN = lic::nn::activation_functions::LINEAR;

typedef lic::nn_models::three_layer_fc::StructureSpecification<DTYPE, INPUT_DIM, LAYER_1_DIM, LAYER_1_FN, LAYER_2_DIM, LAYER_2_FN, OUTPUT_DIM, OUTPUT_FN> NETWORK_STRUCTURE_SPEC;
typedef lic::nn_models::three_layer_fc::AdamSpecification<lic::devices::Generic, NETWORK_STRUCTURE_SPEC, lic::nn::optimizers::adam::DefaultParameters<DTYPE>> NETWORK_SPEC;
typedef lic::nn_models::three_layer_fc::NeuralNetworkAdam<lic::devices::Generic, NETWORK_SPEC> NetworkType;

template <typename T, typename DEVICE, typename SPEC>
T abs_diff(lic::nn::layers::dense::Layer<DEVICE, SPEC> l1, lic::nn::layers::dense::Layer<DEVICE, SPEC> l2) {
    T acc = 0;
    acc += abs_diff_matrix<DTYPE, SPEC::OUTPUT_DIM, SPEC::INPUT_DIM>(l1.weights, l2.weights);
    acc += abs_diff_vector<DTYPE, SPEC::OUTPUT_DIM>(l1.biases, l2.biases);
    return acc;
}
template <typename DEVICE, typename SPEC>
typename SPEC::T abs_diff(lic::nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC> n1, lic::nn_models::three_layer_fc::NeuralNetwork<DEVICE, SPEC> n2) {
    typename SPEC::T acc = 0;
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::LAYER_1::SPEC>(n1.layer_1, n2.layer_1);
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::LAYER_2::SPEC>(n1.layer_2, n2.layer_2);
    acc += abs_diff<DTYPE, DEVICE, typename SPEC::OUTPUT_LAYER::SPEC>(n1.output_layer, n2.output_layer);
    return acc;
}

#include <layer_in_c/utils/persist.h>
TEST(NeuralNetworkPersist, Saving) {

    NetworkType network_1, network_2;
    std::mt19937 rng(2);
    lic::init_weights<NETWORK_SPEC, lic::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(network_1, rng);
    lic::init_weights<NETWORK_SPEC, lic::utils::random::stdlib::uniform<DTYPE, typeof(rng)>, typeof(rng)>(network_2, rng);
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
