#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/memory/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>


#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
//using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

constexpr TI SEQUENCE_LENGTH = 16;
constexpr TI BATCH_SIZE = 32;

using ENVIRONMENT_SPEC = rlt::rl::environments::memory::Specification<T, TI, rlt::rl::environments::memory::DefaultParameters<T, TI>>;
using ENVIRONMENT = rlt::rl::environments::Memory<ENVIRONMENT_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr TI ACTOR_BATCH_SIZE = BATCH_SIZE;
        static constexpr TI CRITIC_BATCH_SIZE = BATCH_SIZE;
    };
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};
#ifdef BENCHMARK
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#else
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS, typename CONTAINER_TYPE_TAG>
struct ConfigApproximatorsSequential{
    template <typename CAPABILITY>
    struct Actor{
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, ENVIRONMENT::Observation::DIM, PARAMETERS::ACTOR_HIDDEN_DIM, rlt::nn::parameters::Gradient, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, PARAMETERS::ACTOR_HIDDEN_DIM, 2*ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  rlt::nn::activation_functions::IDENTITY, typename PARAMETERS::INITIALIZER, CONTAINER_TYPE_TAG, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using ACTOR_TYPE = rlt::nn_models::mlp::BindSpecification<ACTOR_SPEC>;
        using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
        struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, ENVIRONMENT::ACTION_DIM, SAMPLE_AND_SQUASH_LAYER_PARAMETERS, rlt::MatrixDynamicTag, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
        using SAMPLE_AND_SQUASH_MODULE = typename IF::template Module<SAMPLE_AND_SQUASH_LAYER::template Layer>;
        using MODEL = typename IF::template Module<GRU_TEMPLATE::template Layer, typename IF::template Module<ACTOR_TYPE::template NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>>;
    };
    template <typename CAPABILITY>
    struct Critic{
        static constexpr TI INPUT_DIM = ENVIRONMENT::ObservationPrivileged::DIM+ENVIRONMENT::ACTION_DIM;
        using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, ENVIRONMENT::ObservationPrivileged::DIM, PARAMETERS::CRITIC_HIDDEN_DIM, rlt::nn::parameters::groups::Normal, rlt::TensorDynamicTag, true>;
        using GRU_TEMPLATE = rlt::nn::layers::gru::BindSpecification<GRU_SPEC>;
        using SPEC = rlt::nn_models::mlp::Specification<T, TI, PARAMETERS::CRITIC_HIDDEN_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY, typename PARAMETERS::INITIALIZER, CONTAINER_TYPE_TAG, rlt::nn::layers::dense::SequenceInputShapeFactory<TI, SEQUENCE_LENGTH>>;
        using TYPE = rlt::nn_models::mlp::BindSpecification<SPEC>;
        using IF = rlt::nn_models::sequential_v2::Interface<CAPABILITY>;
        using MODEL = typename IF::template Module<GRU_TEMPLATE::template Layer, typename IF::template Module<TYPE::template NeuralNetwork>>;
    };

    using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
    using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
    using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
    using CAPABILITY_ACTOR = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CAPABILITY_CRITIC = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE>;
    using ACTOR_TYPE = typename Actor<CAPABILITY_ACTOR>::MODEL;
    using CRITIC_TYPE = typename Critic<CAPABILITY_CRITIC>::MODEL;
    using CRITIC_TARGET_TYPE = typename Critic<rlt::nn::layer_capability::Forward>::MODEL;
    using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsSequential>;
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CORE_CONFIG>{
    static constexpr TI EVALUATION_EPISODES = 100;
};
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#endif

#ifdef TRAINING
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main(){
    TI seed = 0;
    DEVICE device;
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, 0);
    while(!rlt::step(device, ts)){
#ifndef BENCHMARK
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
#endif
#ifdef BENCHMARK_ABLATION_SIMULATOR
        std::this_thread::sleep_for(std::chrono::duration<T>(8.072980403900147e-05)); // python gymnasium Pendulum-v1 step time
#endif
    }
    rlt::free(device, ts);
    return 0;
}
#else
int main(){
    TI seed = 0;
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, seed);
    using APPROXIMATORS = ConfigApproximatorsSequential<T, TI, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::MatrixDynamicTag>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using ACTOR = APPROXIMATORS::Actor<CAPABILITY>::MODEL;
    using CRITIC = APPROXIMATORS::Critic<CAPABILITY>::MODEL;
    ACTOR actor;
    ACTOR::Buffer<BATCH_SIZE> actor_buffer;
    CRITIC critic;
    CRITIC::Buffer<BATCH_SIZE> critic_buffer;
    CRITIC::CONTENT::Buffer<BATCH_SIZE> critic_gru_buffer;

    std::cout << "Actor input shape: ";
    rlt::print(device, ACTOR::INPUT_SHAPE{});
    std::cout << std::endl;
    std::cout << "Actor output shape: ";
    rlt::print(device, ACTOR::OUTPUT_SHAPE{});
    std::cout << std::endl;

    rlt::malloc(device, actor);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, critic);
    rlt::malloc(device, critic_buffer);
    rlt::malloc(device, critic_gru_buffer);
    rlt::init_weights(device, actor, rng);
    rlt::init_weights(device, critic, rng);

    rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR::INPUT_SHAPE>> actor_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, ACTOR::OUTPUT_SHAPE>> actor_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC::INPUT_SHAPE>> critic_input;
    rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC::CONTENT::OUTPUT_SHAPE>> critic_gru_output;
    rlt::Tensor<rlt::tensor::Specification<T, TI, CRITIC::OUTPUT_SHAPE>> critic_output;
    rlt::malloc(device, actor_input);
    rlt::malloc(device, actor_output);
    rlt::malloc(device, critic_input);
    rlt::malloc(device, critic_gru_output);
    rlt::malloc(device, critic_output);

    rlt::randn(device, critic_input, rng);

    std::cout << "Actual batch size: " << decltype(actor.next_module.content)::ACTUAL_BATCH_SIZE << std::endl;
    std::cout << "Actual batch size layer: " << decltype(actor.next_module.content.output_layer)::SPEC::BATCH_SIZE << std::endl;
    std::cout << "Actual batch size layer: " << decltype(actor.next_module.content.output_layer)::ACTUAL_BATCH_SIZE << std::endl;
    using MLP_OUTPUT = rlt::utils::typing::remove_reference<decltype(rl_tools::output(actor))>::type;
    std::cout << "Actual rows sample and squash output layer: " << MLP_OUTPUT::SPEC::ROWS << std::endl;
    std::cout << "Actual rows mlp output layer: " << decltype(actor.next_module.content.output_layer.output)::ROWS << std::endl;

    auto output_tensor = to_tensor(device, rl_tools::output(actor));
    std::cout << "Output tensor shape: ";
    rlt::print(device, decltype(output_tensor)::SPEC::SHAPE{});
    std::cout << std::endl;
//    auto output_tensor_reshaped = reshape_row_major(device, output_tensor, typename MODULE::OUTPUT_SHAPE{});
//    rlt::forward(device, actor, actor_input, actor_output, actor_buffer, rng);
//    rlt::forward(device, critic, critic_input, critic_output, critic_buffer, rng);
    rlt::forward(device, critic.content, critic_input, critic_gru_buffer, rng);
//    rlt::print(device, critic_input);
//    rlt::print(device, critic_output);
    return 0;
}
#endif


// benchmark training should take < 2s on P1, < 0.75 on M3
