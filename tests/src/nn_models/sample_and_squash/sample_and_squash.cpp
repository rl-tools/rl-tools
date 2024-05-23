#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/activation_functions.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;

using T = float;
using TI = typename DEVICE::index_t;

constexpr TI INPUT_DIM = 10;
constexpr TI OUTPUT_DIM = 5;
constexpr TI NUM_LAYERS = 3;
constexpr TI HIDDEN_DIM = 16;
constexpr auto ACTIVATION_FUNCTION = rlt::nn::activation_functions::TANH;
constexpr TI BATCH_SIZE = 16;
using CONTAINER_TYPE_TAG = rlt::MatrixDynamicTag;

using MLP_SPEC = rlt::nn_models::mlp::Specification<T, TI, INPUT_DIM, 2*OUTPUT_DIM, NUM_LAYERS, HIDDEN_DIM, ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY, CONTAINER_TYPE_TAG>;
using MLP_TYPE = rlt::nn_models::mlp::BindSpecification<MLP_SPEC>;

using SAMPLE_AND_SQUASH_PARAMETERS = rlt::nn::layers::sample_and_squash::DefaultParameters<T>;
using SAMPLE_AND_SQUASH_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, OUTPUT_DIM, SAMPLE_AND_SQUASH_PARAMETERS, rlt::nn::activation_functions::TANH, CONTAINER_TYPE_TAG>;
using SAMPLE_AND_SQUASH = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_SPEC>;

//using SAMPLE_AND_SQUASH_MODULE_SPEC = rlt::nn_models::sequential::Specification<SAMPLE_AND_SQUASH>;
using CAPABILITY_ADAM = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
using IF = rlt::nn_models::sequential::Interface<CAPABILITY_ADAM>;
using SAMPLE_AND_SQUASH_MODULE = IF::Module<SAMPLE_AND_SQUASH::Layer>;
//using ACTOR_SPEC = rlt::nn_models::sequential::Specification<MLP_TYPE, SAMPLE_AND_SQUASH_MODULE>;
using ACTOR = IF::Module<MLP_TYPE::NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>;

int main(){
    ACTOR actor;
    ACTOR::CONTENT::Buffer<BATCH_SIZE> actor_buffer;
    ACTOR::Buffer<BATCH_SIZE> actor_buffer_sequential;
    DEVICE device;

    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 0);

    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE, INPUT_DIM>> input;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE, 2*OUTPUT_DIM>> intermediate_output;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE, OUTPUT_DIM>> output, output_sequential;
    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, actor_buffer_sequential);
    rlt::malloc(device, input);
    rlt::malloc(device, intermediate_output);
    rlt::malloc(device, output);
    rlt::malloc(device, output_sequential);


    rlt::randn(device, input, rng);
    rlt::init_weights(device, actor, rng);

    auto rng2 = rng;
    rlt::evaluate(device, actor, input, output_sequential, actor_buffer_sequential, rng);
    rlt::evaluate(device, actor.content, input, intermediate_output, actor_buffer, rng2);


    T abs_diff = rlt::abs_diff(device, output, output_sequential);

    rlt::print(device, output);
    std::cout << "abs_diff: " << abs_diff << std::endl;

    return 0;
}