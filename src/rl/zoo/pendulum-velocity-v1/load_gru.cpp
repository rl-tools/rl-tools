#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

#include <rl_tools/containers/matrix/persist.h>
#include <rl_tools/containers/tensor/persist.h>
#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/gru/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

namespace rlt = rl_tools;


using DEVICE = rl_tools::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

constexpr TI SEQUENCE_LENGTH = 200;
constexpr TI BATCH_SIZE = 20;
constexpr TI HIDDEN_DIM = 32;
constexpr TI INPUT_DIM = 1;
constexpr TI OUTPUT_DIM = 2;
using GRU_CONFIG = rlt::nn::layers::gru::Configuration<T, TI, HIDDEN_DIM>;
using GRU = rlt::nn::layers::gru::BindConfiguration<GRU_CONFIG>;
using DENSE_CONFIG = rlt::nn::layers::dense::Configuration<T, TI, OUTPUT_DIM, rlt::nn::activation_functions::IDENTITY>;
using DENSE = rlt::nn::layers::dense::BindConfiguration<DENSE_CONFIG>;

template <typename T_CONTENT, typename T_NEXT_MODULE = rlt::nn_models::sequential::OutputModule>
using Module = typename rlt::nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
using MODULE_CHAIN = Module<GRU, Module<DENSE>>;

using CAPABILITY = rlt::nn::capability::Forward<>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, INPUT_DIM>;
using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;



int main(){
    DEVICE device;
    MODEL model;
    MODEL::Buffer<> buffer;
    RNG rng;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, MODEL::OUTPUT_SHAPE>> output, output_target;
    rlt::init(device);
    rlt::malloc(device, rng);
    rlt::malloc(device, model);
    rlt::malloc(device, input);
    rlt::malloc(device, output);
    rlt::malloc(device, output_target);
    rlt::malloc(device, buffer);

    rlt::init(device, rng);
    HighFive::File replay_buffer_file("replay_buffer_gru.h5", HighFive::File::ReadOnly);
    auto model_group = replay_buffer_file.getGroup("model");
    rlt::load(device, model, model_group);
    rlt::load(device, input, replay_buffer_file.getGroup("test"), "input");
    rlt::load(device, output_target, replay_buffer_file.getGroup("test"), "output");
    rlt::evaluate(device, model, input, output, buffer, rng);


    T diff = rlt::abs_diff(device, output, output_target);
    rlt::log(device, device.logger, "Diff: ", diff / decltype(output)::SPEC::SIZE);

    return 0;
}