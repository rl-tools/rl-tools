#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>

namespace rlt = rl_tools;

#include <gtest/gtest.h>


using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = DEVICE::index_t;
constexpr TI INPUT_DIM = 10;
constexpr TI OUTPUT_DIM = 5;
constexpr TI BATCH_SIZE_DEFINITION_MLP = 28;
constexpr TI BATCH_SIZE_DEFINITION = 32;
constexpr TI BATCH_SIZE_OTHER = 30;


template <typename CAPABILITY>
struct Actor{
    using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, INPUT_DIM, OUTPUT_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::BindSpecification<ACTOR_SPEC>;
    using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
    using ACTOR_MODULE = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork>;
    using STANDARDIZATION_LAYER_SPEC = rlt::nn::layers::standardize::Specification<T, TI, INPUT_DIM>;
    using STANDARDIZATION_LAYER = rlt::nn::layers::standardize::BindSpecification<STANDARDIZATION_LAYER_SPEC>;
    using MODEL = typename IF::template Module<STANDARDIZATION_LAYER::template Layer, ACTOR_MODULE>;
};
using CAPABILITY = rlt::nn::layer_capability::Forward;
using ACTOR = Actor<CAPABILITY>::MODEL;

TEST(RL_TOOLS_NN_MODELS_SEQUENTIAL_COMPOSE, MAIN){
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{});

    ACTOR actor;
    ACTOR::Buffer<BATCH_SIZE_DEFINITION> buffer;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE_OTHER, INPUT_DIM>> input;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, BATCH_SIZE_OTHER, OUTPUT_DIM>> output;

    rlt::malloc(device, actor);
    rlt::malloc(device, buffer);

    rlt::init_weights(device, actor, rng);
    rlt::evaluate(device, actor, input, output, buffer, rng);

    rlt::free(device, actor);
}
