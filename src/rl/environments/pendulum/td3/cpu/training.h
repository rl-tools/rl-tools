// ------------ Groups 1 ------------
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
#include <backprop_tools/operations/cpu_tensorboard/group_1.h>
#endif
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_1.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_1.h>
#else
#include <backprop_tools/operations/cpu/group_1.h>
#endif
#endif
// ------------ Groups 2 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_2.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_2.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_2.h>
#else
#include <backprop_tools/operations/cpu/group_2.h>
#endif
#endif
// ------------ Groups 3 ------------
#include <backprop_tools/operations/cpu_tensorboard/group_3.h>
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/operations/cpu_mkl/group_3.h>
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/operations/cpu_accelerate/group_3.h>
#else
#include <backprop_tools/operations/cpu/group_3.h>
#endif
#endif

namespace bpt = backprop_tools;

#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU_TENSORBOARD>;
#else
using DEV_SPEC = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, bpt::devices::logging::CPU>;
#endif

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL
#include <backprop_tools/nn/operations_cpu_mkl.h>
using DEVICE = bpt::devices::CPU_MKL<DEV_SPEC>;
#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/nn/operations_cpu_accelerate.h>
using DEVICE = bpt::devices::CPU_ACCELERATE<DEV_SPEC>;
#else
#include <backprop_tools/nn/operations_generic.h>
using DEVICE = bpt::devices::CPU<DEV_SPEC>;
#endif
#endif

#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/operations_generic.h>


#include <backprop_tools/rl/utils/evaluation.h>

#include <filesystem>


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <backprop_tools/rl/environments/pendulum/ui.h>
#include <backprop_tools/rl/utils/evaluation_visual.h>
#endif


#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif


using DTYPE = float;

typedef bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef bpt::rl::environments::pendulum::UI<DTYPE> UI;
#endif

struct DEVICE_SPEC: bpt::devices::DefaultCPUSpecification {
#if defined(BACKPROP_TOOLS_ENABLE_TENSORBOARD) && !defined(BACKPROP_TOOLS_DISABLE_TENSORBOARD)
    using LOGGING = bpt::devices::logging::CPU_TENSORBOARD;
#else
    using LOGGING = bpt::devices::logging::CPU;
#endif
};
struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters;

using ActorStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = bpt::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ActorCriticType = bpt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
constexpr DEVICE::index_t STEP_LIMIT = 1000;
#else
constexpr DEVICE::index_t STEP_LIMIT = 10000;
#endif

constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = STEP_LIMIT;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        bpt::rl::components::off_policy_runner::DefaultParameters<DTYPE>
>;
bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

void run(){
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    DEVICE nn_dev;
    nn_dev.logger = &logger;

    OPTIMIZER optimizer;

    std::mt19937 rng(4);
    bpt::malloc(nn_dev, actor_critic);
    bpt::init(nn_dev, actor_critic, optimizer, rng);

    bool ui = false;

    bpt::construct(device, device.logger);

    bpt::malloc(device, off_policy_runner);
    ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    bpt::init(device, off_policy_runner, envs);

    bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::BuffersForwardBackward<ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    bpt::malloc(device, critic_batch);
    bpt::malloc(device, critic_training_buffers);
    bpt::malloc(device, critic_buffers[0]);
    bpt::malloc(device, critic_buffers[1]);

    bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    bpt::malloc(device, actor_batch);
    bpt::malloc(device, actor_training_buffers);
    bpt::malloc(device, actor_buffers_eval);
    bpt::malloc(device, actor_buffers[0]);
    bpt::malloc(device, actor_buffers[1]);

    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;
    bpt::malloc(device, observations_mean);
    bpt::malloc(device, observations_std);
    bpt::set_all(device, observations_mean, 0);
    bpt::set_all(device, observations_std, 1);


    auto start_time = std::chrono::high_resolution_clock::now();

    for(int step_i = 0; step_i < STEP_LIMIT; step_i+=OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        bpt::set_step(device, device.logger, step_i);
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, decltype(actor_critic.actor), decltype(actor_critic.critic_1)>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
        }
#endif
        bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
        bpt::set_state(ui, off_policy_runner.state);
#endif

        if(step_i > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - start_time;
                std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
            }

            for(int critic_i = 0; critic_i < 2; critic_i++){
                bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                bpt::gather_batch(device, off_policy_runner, critic_batch, rng);
                bpt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

//            DTYPE critic_1_loss = bpt::train_critic(device, actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
//            bpt::train_critic(device, actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                {
                    bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    bpt::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                bpt::update_critic_targets(device, actor_critic);
                bpt::update_actor_target(device, actor_critic);
            }
        }
#ifndef BACKPROP_TOOLS_DISABLE_EVALUATION
        if(step_i % 1000 == 0){
//            auto result = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<1, ENVIRONMENT_STEP_LIMIT>(), rng, true);
            auto result = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<10, ENVIRONMENT_STEP_LIMIT>(), observations_mean, observations_std, rng);
            std::cout << "Mean return: " << result.mean << std::endl;
            bpt::add_scalar(device, device.logger, "mean_return", result.mean);
//            if(step_i >= 6000){
//                ASSERT_GT(mean_return, -1000);
//            }
//            if(step_i >= 14000){
//                ASSERT_GT(mean_return, -400);
//            }

//#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
//            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
//#endif
#ifdef BACKPROP_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
            //            for(int evaluation_i = 0; evaluation_i < 10; evaluation_i++){
//                ENVIRONMENT::State initial_state;
//                bpt::sample_initial_state(env, initial_state, rng);
//                bpt::evaluate_visual<ENVIRONMENT, UI, ActorCriticType::ACTOR_NETWORK_TYPE, ENVIRONMENT_STEP_LIMIT, 5>(env, ui, actor_critic.actor, initial_state);
//            }
#endif
        }
#endif
    }
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    }
    bpt::free(device, critic_batch);
    bpt::free(device, critic_training_buffers);
    bpt::free(device, actor_batch);
    bpt::free(device, actor_training_buffers);
    bpt::free(device, off_policy_runner);
    bpt::free(device, actor_critic);
    bpt::free(device, observations_mean);
    bpt::free(device, observations_std);

    bpt::destruct(device, device.logger);
}

