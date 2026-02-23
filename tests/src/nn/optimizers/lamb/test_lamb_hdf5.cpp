#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/lamb/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/lamb/operations_generic.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "../../../utils/utils.h"

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <sstream>
#include <cmath>
#include <iostream>

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

constexpr TI INPUT_DIM = 4;
constexpr TI HIDDEN_DIM = 32;
constexpr TI OUTPUT_DIM = 4;

using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 1, INPUT_DIM>;
using NETWORK_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, 2, HIDDEN_DIM, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;
using OPTIMIZER_SPEC = rlt::nn::optimizers::lamb::Specification<TYPE_POLICY, TI>;
using OPTIMIZER = rlt::nn::optimizers::Lamb<OPTIMIZER_SPEC>;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using NetworkType = rlt::nn_models::mlp::NeuralNetwork<NETWORK_CONFIG, CAPABILITY, INPUT_SHAPE>;

template <typename SPEC>
T compare_2d(DEVICE& device, rlt::Tensor<SPEC>& tensor, HighFive::Group& group, const std::string& name) {
    std::vector<std::vector<T>> ref;
    group.getDataSet(name).read(ref);
    auto view = rlt::matrix_view(device, tensor);
    return abs_diff_matrix(view, ref);
}

template <typename SPEC>
T compare_1d(DEVICE& device, rlt::Tensor<SPEC>& tensor, HighFive::Group& group, const std::string& name) {
    std::vector<T> ref;
    group.getDataSet(name).read(ref);
    T acc = 0;
    for(TI i = 0; i < ref.size(); i++){
        acc += std::abs(rlt::get(device, tensor, i) - ref[i]);
    }
    return acc;
}

template <typename SPEC>
void load_2d(DEVICE& device, rlt::Tensor<SPEC>& tensor, HighFive::Group& group, const std::string& name) {
    std::vector<std::vector<T>> data;
    group.getDataSet(name).read(data);
    auto view = rlt::matrix_view(device, tensor);
    assign(data, view);
}

template <typename SPEC>
void load_1d(DEVICE& device, rlt::Tensor<SPEC>& tensor, HighFive::Group& group, const std::string& name) {
    std::vector<T> data;
    group.getDataSet(name).read(data);
    for(TI i = 0; i < data.size(); i++){
        rlt::set(device, tensor, data[i], i);
    }
}

T compare_layer_weights(DEVICE& device, NetworkType& network, HighFive::Group& group, const std::string& layer_name, bool input_layer) {
    auto layer_group = group.getGroup(layer_name);
    T diff = 0;
    if(input_layer){
        diff += compare_2d(device, network.input_layer.weights.parameters, layer_group, "weight");
        diff += compare_1d(device, network.input_layer.biases.parameters, layer_group, "bias");
    } else {
        diff += compare_2d(device, network.output_layer.weights.parameters, layer_group, "weight");
        diff += compare_1d(device, network.output_layer.biases.parameters, layer_group, "bias");
    }
    return diff;
}

T compare_layer_grads(DEVICE& device, NetworkType& network, HighFive::Group& group, const std::string& layer_name, bool input_layer) {
    auto layer_group = group.getGroup(layer_name);
    T diff = 0;
    if(input_layer){
        diff += compare_2d(device, network.input_layer.weights.gradient, layer_group, "weight");
        diff += compare_1d(device, network.input_layer.biases.gradient, layer_group, "bias");
    } else {
        diff += compare_2d(device, network.output_layer.weights.gradient, layer_group, "weight");
        diff += compare_1d(device, network.output_layer.biases.gradient, layer_group, "bias");
    }
    return diff;
}

T compare_layer_optimizer_state(DEVICE& device, NetworkType& network, HighFive::Group& group, const std::string& layer_name, bool input_layer) {
    auto layer_group = group.getGroup(layer_name);
    T diff = 0;
    if(input_layer){
        auto w_group = layer_group.getGroup("weight");
        auto b_group = layer_group.getGroup("bias");
        diff += compare_2d(device, network.input_layer.weights.gradient_first_order_moment, w_group, "exp_avg");
        diff += compare_2d(device, network.input_layer.weights.gradient_second_order_moment, w_group, "exp_avg_sq");
        diff += compare_1d(device, network.input_layer.biases.gradient_first_order_moment, b_group, "exp_avg");
        diff += compare_1d(device, network.input_layer.biases.gradient_second_order_moment, b_group, "exp_avg_sq");
    } else {
        auto w_group = layer_group.getGroup("weight");
        auto b_group = layer_group.getGroup("bias");
        diff += compare_2d(device, network.output_layer.weights.gradient_first_order_moment, w_group, "exp_avg");
        diff += compare_2d(device, network.output_layer.weights.gradient_second_order_moment, w_group, "exp_avg_sq");
        diff += compare_1d(device, network.output_layer.biases.gradient_first_order_moment, b_group, "exp_avg");
        diff += compare_1d(device, network.output_layer.biases.gradient_second_order_moment, b_group, "exp_avg_sq");
    }
    return diff;
}

TEST(RL_TOOLS_NN_OPTIMIZERS_LAMB, COMPARE_WITH_TIMM){
    DEVICE device;

    std::string data_file_path = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/lamb_test_data.h5";
    auto file = HighFive::File(data_file_path, HighFive::File::ReadOnly);

    auto config = file.getGroup("config");
    ASSERT_EQ(config.getAttribute("input_dim").read<int>(), INPUT_DIM);
    ASSERT_EQ(config.getAttribute("hidden_dim").read<int>(), HIDDEN_DIM);
    ASSERT_EQ(config.getAttribute("output_dim").read<int>(), OUTPUT_DIM);
    int n_steps = config.getAttribute("n_steps").read<int>();
    T lr = config.getAttribute("lr").read<T>();
    T beta1 = config.getAttribute("beta1").read<T>();
    T beta2 = config.getAttribute("beta2").read<T>();
    T eps = config.getAttribute("eps").read<T>();

    OPTIMIZER optimizer;
    NetworkType network;
    typename NetworkType::Buffer<> buffers;
    rlt::malloc(device, optimizer);
    rlt::malloc(device, network);
    rlt::malloc(device, buffers);
    rlt::init(device, optimizer);

    auto& opt_params = rlt::get_ref(device, optimizer.parameters, 0);
    opt_params.alpha = lr;
    opt_params.beta_1 = beta1;
    opt_params.beta_2 = beta2;
    opt_params.epsilon = eps;
    opt_params.epsilon_sqrt = 0;

    auto init_group = file.getGroup("init");
    auto init_input = init_group.getGroup("input_layer");
    auto init_output = init_group.getGroup("output_layer");
    load_2d(device, network.input_layer.weights.parameters, init_input, "weight");
    load_1d(device, network.input_layer.biases.parameters, init_input, "bias");
    load_2d(device, network.output_layer.weights.parameters, init_output, "weight");
    load_1d(device, network.output_layer.biases.parameters, init_output, "bias");

    rlt::reset_optimizer_state(device, optimizer, network);

    std::vector<std::vector<T>> input_data_2d;
    file.getDataSet("input").read(input_data_2d);
    std::vector<std::vector<T>> target_data_2d;
    file.getDataSet("target").read(target_data_2d);

    T input_buf[INPUT_DIM];
    T target_buf[OUTPUT_DIM];
    for(TI i = 0; i < INPUT_DIM; i++) input_buf[i] = input_data_2d[0][i];
    for(TI i = 0; i < OUTPUT_DIM; i++) target_buf[i] = target_data_2d[0][i];

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, INPUT_DIM>> input_matrix;
    input_matrix._data = input_buf;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> target_matrix;
    target_matrix._data = target_buf;

    auto steps_group = file.getGroup("steps");

    for(int step_i = 0; step_i < n_steps; step_i++){
        std::stringstream ss;
        ss << step_i;
        auto step_group = steps_group.getGroup(ss.str());

        rlt::zero_gradient(device, network);
        bool rng_flag = false;
        rlt::forward(device, network, input_matrix, buffers, rng_flag);

        T ref_loss = step_group.getAttribute("loss").read<T>();
        T loss = rlt::nn::loss_functions::mse::evaluate(device, network.output_layer.output, target_matrix);
        T loss_diff = std::abs(loss - ref_loss);
        std::cout << "Step " << step_i << ": loss=" << loss << " ref_loss=" << ref_loss << " diff=" << loss_diff << std::endl;
        ASSERT_LT(loss_diff, 1e-12) << "Loss mismatch at step " << step_i;

        T d_loss_d_output_buf[OUTPUT_DIM];
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OUTPUT_DIM>> d_loss_d_output;
        d_loss_d_output._data = d_loss_d_output_buf;
        rlt::nn::loss_functions::mse::gradient(device, network.output_layer.output, target_matrix, d_loss_d_output);
        rlt::backward(device, network, input_matrix, d_loss_d_output, buffers);

        auto grads_group = step_group.getGroup("gradients");
        T grad_diff = 0;
        grad_diff += compare_layer_grads(device, network, grads_group, "input_layer", true);
        grad_diff += compare_layer_grads(device, network, grads_group, "output_layer", false);
        std::cout << "  gradient diff: " << grad_diff << std::endl;
        ASSERT_LT(grad_diff, 1e-12) << "Gradient mismatch at step " << step_i;

        rlt::step(device, optimizer, network);

        auto weights_group = step_group.getGroup("weights");
        T weight_diff = 0;
        weight_diff += compare_layer_weights(device, network, weights_group, "input_layer", true);
        weight_diff += compare_layer_weights(device, network, weights_group, "output_layer", false);
        std::cout << "  weight diff: " << weight_diff << std::endl;
        ASSERT_LT(weight_diff, 1e-10) << "Weight mismatch at step " << step_i;

        auto opt_group = step_group.getGroup("optimizer_state");
        T opt_diff = 0;
        opt_diff += compare_layer_optimizer_state(device, network, opt_group, "input_layer", true);
        opt_diff += compare_layer_optimizer_state(device, network, opt_group, "output_layer", false);
        std::cout << "  optimizer state diff: " << opt_diff << std::endl;
        ASSERT_LT(opt_diff, 1e-12) << "Optimizer state mismatch at step " << step_i;
    }

    rlt::free(device, buffers);
    rlt::free(device, network);
    rlt::free(device, optimizer);
}
