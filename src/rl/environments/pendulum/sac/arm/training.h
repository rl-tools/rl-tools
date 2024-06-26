
//#define RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
//#include <rl_tools/operations/arm.h>
//#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
//#include <rl_tools/logging/operations_arduino.h>
//#else
//#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
//#include <rl_tools/operations/cpu.h>
//#endif
//#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
//#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
//#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
//#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
//#include <rl_tools/nn_models/mlp/operations_generic.h>
//#include <rl_tools/nn_models/sequential/operations_generic.h>
//#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
#include <rl_tools/operations/arm.h>
#include <rl_tools/logging/operations_arduino.h>
#else
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#include <rl_tools/version.h>
#include <rl_tools/rl_tools.h>
#include <rl_tools/operations/arm/group_1.h>
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/arm/group_2.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/arm/group_3.h>
#include <rl_tools/operations/cpu/group_3.h>
#endif

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>


#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>


//#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
//#include <rl_tools/rl/loop/steps/evaluation/config.h>
//#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
//#include <rl_tools/rl/loop/steps/timing/config.h>
//#endif
//#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
//#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
//#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
//#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>
//#endif

namespace rlt = rl_tools;

#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
using LOGGING = rlt::devices::logging::ARDUINO;
using DEVICE = rlt::devices::arm::OPT<rlt::devices::arm::Specification<rlt::devices::math::ARM, rlt::devices::random::ARM, LOGGING>>;
#else
using DEVICE = rlt::devices::DefaultCPU;
#endif

using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;
using CONTAINER_TYPE_TAG = rlt::MatrixStaticTag;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<T, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr TI ACTOR_BATCH_SIZE = 100;
        static constexpr TI CRITIC_BATCH_SIZE = 100;
    };
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};
template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS, typename CONTAINER_TYPE_TAG>
struct APPROXIMATOR_CONFIG{
    using ACTOR_CONTAINER_TYPE_TAG = rlt::MatrixDynamicTag;
//    using ACTOR_CONTAINER_TYPE_TAG = rlt::MatrixStaticTag;
    using CRITIC_CONTAINER_TYPE_TAG = rlt::MatrixStaticTag;
    template <typename CAPABILITY>
    struct Actor{
        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, 2*ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  rlt::nn::activation_functions::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, ACTOR_CONTAINER_TYPE_TAG>;
        using ACTOR_TYPE = rlt::nn_models::mlp::BindSpecification<ACTOR_SPEC>;
        using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
        struct SAMPLE_AND_SQUASH_LAYER_PARAMETERS{
            static constexpr T LOG_STD_LOWER_BOUND = PARAMETERS::LOG_STD_LOWER_BOUND;
            static constexpr T LOG_STD_UPPER_BOUND = PARAMETERS::LOG_STD_UPPER_BOUND;
            static constexpr T LOG_PROBABILITY_EPSILON = PARAMETERS::LOG_PROBABILITY_EPSILON;
            static constexpr bool ADAPTIVE_ALPHA = PARAMETERS::ADAPTIVE_ALPHA;
            static constexpr T ALPHA = PARAMETERS::ALPHA;
            static constexpr T TARGET_ENTROPY = PARAMETERS::TARGET_ENTROPY;
        };
        using SAMPLE_AND_SQUASH_LAYER_SPEC = rlt::nn::layers::sample_and_squash::Specification<T, TI, ENVIRONMENT::ACTION_DIM, SAMPLE_AND_SQUASH_LAYER_PARAMETERS, ACTOR_CONTAINER_TYPE_TAG>;
        using SAMPLE_AND_SQUASH_LAYER = rlt::nn::layers::sample_and_squash::BindSpecification<SAMPLE_AND_SQUASH_LAYER_SPEC>;
        using SAMPLE_AND_SQUASH_MODULE = typename IF::template Module<SAMPLE_AND_SQUASH_LAYER::template Layer>;
        using MODEL = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork, SAMPLE_AND_SQUASH_MODULE>;
    };
    template <typename CAPABILITY>
    struct Critic{
        static constexpr TI INPUT_DIM = ENVIRONMENT::Observation::DIM+ENVIRONMENT::ACTION_DIM;
        using SPEC = rlt::nn_models::mlp::Specification<T, TI, INPUT_DIM, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, rlt::nn::activation_functions::IDENTITY, rlt::nn::layers::dense::DefaultInitializer<T, TI>, CRITIC_CONTAINER_TYPE_TAG>;
        using TYPE = rlt::nn_models::mlp::BindSpecification<SPEC>;
        using IF = rlt::nn_models::sequential::Interface<CAPABILITY>;
        using MODEL = typename IF::template Module<TYPE::template NeuralNetwork>;
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

using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, APPROXIMATOR_CONFIG, CONTAINER_TYPE_TAG>;
#ifdef BENCHMARK
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#else
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
#else
template <typename NEXT>
struct EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 1000;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
#ifdef BENCHMARK
#endif
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#else
using LOOP_CONFIG = LOOP_EVAL_CONFIG;
#endif
#endif

using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

template <typename DEVICE, typename LOOP_STATE>
void print_sizes(DEVICE& device, LOOP_STATE& ts){
    rlt::log(device, device.logger, "ActorCritic size: ", sizeof(ts.actor_critic));
    rlt::log(device, device.logger, "ActorCritic.actor size: ", sizeof(ts.actor_critic.actor));
    rlt::log(device, device.logger, "ActorCritic.critic_1 size: ", sizeof(ts.actor_critic.critic_1));
    rlt::log(device, device.logger, "ActorCritic.critic_2 size: ", sizeof(ts.actor_critic.critic_2));
    rlt::log(device, device.logger, "ActorCritic.critic_target_1 size: ", sizeof(ts.actor_critic.critic_target_1));
    rlt::log(device, device.logger, "ActorCritic.critic_target_2 size: ", sizeof(ts.actor_critic.critic_target_2));
    rlt::log(device, device.logger, "OffPolicyRunner size: ", sizeof(ts.off_policy_runner));
    rlt::log(device, device.logger, "OffPolicyRunner.replay_buffers size: ", sizeof(ts.off_policy_runner.replay_buffers));
    rlt::log(device, device.logger, "CriticBatch size: ", sizeof(ts.critic_batch));
    rlt::log(device, device.logger, "CriticTrainingBuffers size: ", sizeof(ts.critic_training_buffers));
    rlt::log(device, device.logger, "CriticBuffers size: ", sizeof(ts.critic_buffers));
    rlt::log(device, device.logger, "ActorBatch size: ", sizeof(ts.actor_batch));
    rlt::log(device, device.logger, "ActorTrainingBuffers size: ", sizeof(ts.actor_training_buffers));
    rlt::log(device, device.logger, "ActorBuffers size: ", sizeof(ts.actor_buffers));
    rlt::log(device, device.logger, "Total: ", sizeof(ts.actor_critic) + sizeof(ts.off_policy_runner) + sizeof(ts.critic_batch) + sizeof(ts.critic_training_buffers) + sizeof(ts.critic_buffers) + sizeof(ts.actor_batch) + sizeof(ts.actor_training_buffers) + sizeof(ts.actor_buffers));
}

void train(){
    DEVICE device;
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, 0);
    print_sizes(device, ts);
    while(!rlt::step(device, ts)){
    }
    rlt::free(device, ts);
}


// benchmark training should take < 2s on P1, < 0.75 on M3
