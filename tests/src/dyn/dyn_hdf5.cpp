#include <rl_tools/operations/cpu.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/resnet/resnet.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/mlp/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <rl_tools/dyn/persist.h>

#define RL_TOOLS_STRINGIZE(x) #x
#define RL_TOOLS_MACRO_TO_STR(macro) RL_TOOLS_STRINGIZE(macro)

namespace rlt = rl_tools;

#include <iostream>
#include <cmath>

using DEVICE = rl_tools::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

#include <gtest/gtest.h>

TEST(TEST_DYN_HDF5, dense_layer){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    constexpr TI INPUT_DIM = 15, OUTPUT_DIM = 10;
    using CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, 3, INPUT_DIM>;
    using SPEC = rlt::nn::layers::dense::Specification<CONFIG, rlt::nn::capability::Forward<>, INPUT_SHAPE>;
    rlt::nn::layers::dense::LayerForward<SPEC> layer;
    rlt::malloc(device, layer);
    rlt::init_weights(device, layer, rng);
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename SPEC::OUTPUT_SHAPE>> output_static;
    rlt::malloc(device, input); rlt::malloc(device, output_static);
    rlt::randn(device, input, rng);
    rlt::nn::layers::dense::Buffer buffer_static;
    rlt::evaluate(device, layer, input, output_static, buffer_static, rng);
    // Save to HDF5
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_hdf5_dense.h5";
    {
        auto file = HighFive::File(dp, HighFive::File::Overwrite);
        auto group = rlt::create_group(device, file, "layer");
        rlt::save(device, layer, group);
    }
    // Load into dyn
    auto file = HighFive::File(dp, HighFive::File::ReadOnly);
    auto group = rlt::get_group(device, file, "layer");
    rlt::dyn::Layer<TI> dyn_layer;
    rlt::load(device, dyn_layer, group);
    ASSERT_EQ(dyn_layer.type, rlt::dyn::LayerType::DENSE);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> dyn_input, dyn_output;
    TI in_shape[] = {(TI)3, INPUT_DIM};
    rlt::dyn::set_shape(dyn_input, (TI)2, in_shape);
    dyn_input.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_input);
    auto input_mat = rlt::matrix_view(device, input);
    for(TI i = 0; i < 3; i++) for(TI j = 0; j < INPUT_DIM; j++) rlt::dyn::set(device, dyn_input, i * INPUT_DIM + j, rlt::get(input_mat, i, j));
    TI output_shape[] = {(TI)3, OUTPUT_DIM};
    rlt::dyn::set_shape(dyn_output, (TI)2, output_shape);
    dyn_output.type = rlt::dyn::Type::FLOAT32;
    rlt::malloc(device, dyn_output);
    rlt::dyn::propagate_shapes(dyn_layer, dyn_input.shape, dyn_input.rank, dyn_input.size);
    rlt::dyn::Buffer<TI> dyn_buffer;
    dyn_buffer.layer = &dyn_layer;
    rlt::malloc(device, dyn_buffer);
    rlt::evaluate(device, dyn_layer, dyn_input, dyn_output, dyn_buffer);
    auto output_mat = rlt::matrix_view(device, output_static);
    T max_diff = 0;
    for(TI i = 0; i < 3; i++) for(TI j = 0; j < OUTPUT_DIM; j++){
        T diff = std::abs(rlt::get(output_mat, i, j) - rlt::dyn::get(device, dyn_output, i * OUTPUT_DIM + j));
        if(diff > max_diff) max_diff = diff;
    }
    std::cout << "HDF5 Dense max diff: " << max_diff << std::endl;
    ASSERT_NEAR(max_diff, 0, 1e-5);
    rlt::free(device, dyn_input); rlt::free(device, dyn_output); rlt::free(device, dyn_buffer);
    rlt::free(device, dyn_layer); rlt::free(device, layer); rlt::free(device, input); rlt::free(device, output_static);
}

TEST(TEST_DYN_HDF5, sequential_dense_gru_mlp){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);
    using INPUT_LAYER_CONFIG = rlt::nn::layers::dense::Configuration<TYPE_POLICY, TI, 10, rlt::nn::activation_functions::ActivationFunction::RELU>;
    using GRU_LAYER_CONFIG = rlt::nn::layers::gru::Configuration<TYPE_POLICY, TI, 20>;
    using MLP_LAYER_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 15, 3, 7, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::ActivationFunction::IDENTITY>;
    using CHAIN = rlt::nn_models::sequential::Module<
        rlt::nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>,
        rlt::nn_models::sequential::Module<rlt::nn::layers::gru::BindConfiguration<GRU_LAYER_CONFIG>,
        rlt::nn_models::sequential::Module<rlt::nn_models::mlp::BindConfiguration<MLP_LAYER_CONFIG>>>>;
    constexpr TI SEQ_LEN = 5, BATCH_SIZE = 3, INPUT_DIM = 10, OUTPUT_DIM = 15;
    using MODEL = rlt::nn_models::sequential::Build<rlt::nn::capability::Forward<>, CHAIN, rlt::tensor::Shape<TI, SEQ_LEN, BATCH_SIZE, INPUT_DIM>>;
    MODEL model; MODEL::Buffer<> buf;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::INPUT_SHAPE>> in;
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>> out;
    rlt::malloc(device, model); rlt::malloc(device, buf); rlt::malloc(device, in); rlt::malloc(device, out);
    rlt::init_weights(device, model, rng); rlt::randn(device, in, rng);
    rlt::evaluate(device, model, in, out, buf, rng);
    std::string dp = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/test_dyn_hdf5_sequential.h5";
    { auto file = HighFive::File(dp, HighFive::File::Overwrite); auto g = rlt::create_group(device, file, "model"); rlt::save(device, model, g); }
    auto file = HighFive::File(dp, HighFive::File::ReadOnly);
    auto g = rlt::get_group(device, file, "model");
    rlt::dyn::Layer<TI> dm;
    rlt::load(device, dm, g);
    ASSERT_EQ(dm.type, rlt::dyn::LayerType::SEQUENTIAL);
    rlt::dyn::Tensor<rlt::dyn::TensorSpecification<TI>> di, d_out;
    TI dis[] = {SEQ_LEN, BATCH_SIZE, INPUT_DIM};
    rlt::dyn::set_shape(di, (TI)3, dis); di.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, di);
    auto im = rlt::matrix_view(device, in);
    for(TI i = 0; i < SEQ_LEN * BATCH_SIZE; i++) for(TI j = 0; j < INPUT_DIM; j++) rlt::dyn::set(device, di, i * INPUT_DIM + j, rlt::get(im, i, j));
    TI dos[] = {SEQ_LEN, BATCH_SIZE, OUTPUT_DIM}; rlt::dyn::set_shape(d_out, (TI)3, dos); d_out.type = rlt::dyn::Type::FLOAT32; rlt::malloc(device, d_out);
    rlt::dyn::propagate_shapes(dm, di.shape, di.rank, di.size);
    rlt::dyn::Buffer<TI> db; db.layer = &dm; rlt::malloc(device, db);
    rlt::evaluate(device, dm, di, d_out, db);
    auto om = rlt::matrix_view(device, out);
    T md = 0;
    for(TI i = 0; i < SEQ_LEN * BATCH_SIZE; i++) for(TI j = 0; j < OUTPUT_DIM; j++){
        T d = std::abs(rlt::get(om, i, j) - rlt::dyn::get(device, d_out, i * OUTPUT_DIM + j));
        if(d > md) md = d;
    }
    std::cout << "HDF5 Sequential max diff: " << md << std::endl;
    ASSERT_NEAR(md, 0, 1e-5);
    rlt::free(device, di); rlt::free(device, d_out); rlt::free(device, db); rlt::free(device, dm);
    rlt::free(device, model); rlt::free(device, buf); rlt::free(device, in); rlt::free(device, out);
}
