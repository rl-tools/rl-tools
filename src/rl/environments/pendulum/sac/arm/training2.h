#define RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
#include <rl_tools/operations/arm.h>
#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
#include <rl_tools/logging/operations_arduino.h>
#else
#define RL_TOOLS_DEVICES_DISABLE_REDEFINITION_DETECTION
#include <rl_tools/operations/cpu.h>
#endif
#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>


#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
#include <rl_tools/rl/loop/steps/timing/config.h>
#endif
#include <rl_tools/rl/algorithms/sac/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
#include <rl_tools/rl/loop/steps/timing/operations.h>
#endif

namespace rlt = rl_tools;

#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
using LOGGING = rlt::devices::logging::ARDUINO;
#else
using LOGGING = rlt::devices::logging::CPU;
#endif

using DEVICE = rlt::devices::ARM<rlt::devices::arm::Specification<rlt::devices::math::ARM, rlt::devices::random::ARM, LOGGING>>;
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
#ifdef BENCHMARK
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsSequential, CONTAINER_TYPE_TAG>;
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#else
using LOOP_CONFIG = LOOP_CORE_CONFIG;
#endif
#else
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsMLP, CONTAINER_TYPE_TAG>;
template <typename T, typename TI, typename NEXT>
//struct EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
//    static constexpr TI EVALUATION_INTERVAL = 100;
//};
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG>;
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
#ifndef BENCHMARK
        if(ts.step == 5000){
//            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
#endif
    }
    rlt::free(device, ts);
}


// benchmark training should take < 2s on P1, < 0.75 on M3
