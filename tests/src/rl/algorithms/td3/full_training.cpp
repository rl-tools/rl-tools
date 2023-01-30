#include <layer_in_c/operations/cpu_tensorboard.h>

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
#include <layer_in_c/operations/cpu_mkl.h>
#include <layer_in_c/nn/operations_cpu_mkl.h>
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/operations/cpu_accelerate.h>
#include <layer_in_c/nn/operations_cpu_accelerate.h>
#else
#include <layer_in_c/operations/cpu.h>
#include <layer_in_c/nn/operations_generic.h>
#endif
#endif

#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/operations_generic.h>


#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <filesystem>


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
#include <layer_in_c/rl/environments/pendulum/ui.h>
#include <layer_in_c/rl/utils/evaluation_visual.h>
#endif


#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
#include "plot_policy_and_value_function.h"
#endif


namespace lic = layer_in_c;
using DTYPE = float;

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL
using DEVICE = lic::devices::CPU_MKL<lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>>;
#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
using DEVICE = lic::devices::CPU_ACCELERATE<lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>>;
#else
using DEVICE = lic::devices::CPU<lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>>;
#endif
#endif


typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
typedef lic::rl::environments::pendulum::UI<DTYPE> UI;
#endif
ENVIRONMENT env;

struct DEVICE_SPEC: lic::devices::DefaultCPUSpecification {
    using LOGGING = lic::devices::logging::CPU;
};
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters;

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
DEVICE::SPEC::LOGGING logger;
DEVICE ac_dev(logger);
//DEVICE::SPEC::LOGGING nn_logger;
DEVICE nn_dev(logger);
lic::rl::components::OffPolicyRunner<
    lic::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
    >
> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(LAYER_IN_C_RL_ALGORITHMS_TD3_FULL_TRAINING, TEST_FULL_TRAINING) {
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
    UI ui;
#endif
    std::mt19937 rng(4);
    lic::malloc(nn_dev, actor_critic);
    lic::init(nn_dev, actor_critic, rng);

    bool ui = false;

    lic::construct(ac_dev.logger);

    lic::malloc(ac_dev, off_policy_runner);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::Buffers<> critic_buffers[2];
    lic::malloc(ac_dev, critic_batch);
    lic::malloc(ac_dev, critic_training_buffers);
    lic::malloc(ac_dev, critic_buffers[0]);
    lic::malloc(ac_dev, critic_buffers[1]);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<> actor_buffers[2];
    lic::malloc(ac_dev, actor_batch);
    lic::malloc(ac_dev, actor_training_buffers);
    lic::malloc(ac_dev, actor_buffers[0]);
    lic::malloc(ac_dev, actor_buffers[1]);


    auto start_time = std::chrono::high_resolution_clock::now();

    for(int step_i = 0; step_i < 15000; step_i++){
        ac_dev.logger.step = step_i;
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
        if(step_i % 20 == 0){
            plot_policy_and_value_function<DTYPE, ENVIRONMENT, decltype(actor_critic.actor), decltype(actor_critic.critic_1)>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
        }
#endif
        lic::step(ac_dev, off_policy_runner, actor_critic.actor, rng);
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
        lic::set_state(ui, off_policy_runner.state);
#endif

        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - start_time;
                std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
            }

            for(int critic_i = 0; critic_i < 2; critic_i++){
                lic::target_action_noise(ac_dev, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                lic::gather_batch(ac_dev, off_policy_runner.replay_buffer, critic_batch, rng);
                lic::train_critic(ac_dev, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

//            DTYPE critic_1_loss = lic::train_critic(ac_dev, actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
//            lic::train_critic(ac_dev, actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                {
                    lic::gather_batch(ac_dev, off_policy_runner.replay_buffer, actor_batch, rng);
                    lic::train_actor(ac_dev, actor_critic, actor_batch, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                lic::update_critic_targets(ac_dev, actor_critic);
                lic::update_actor_target(ac_dev, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(ui), decltype(actor_critic.actor), decltype(rng), ENVIRONMENT_STEP_LIMIT, true>(ac_dev, env, ui, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
//            if(step_i >= 6000){
//                ASSERT_GT(mean_return, -1000);
//            }
//            if(step_i >= 14000){
//                ASSERT_GT(mean_return, -400);
//            }

//#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_OUTPUT_PLOTS
//            plot_policy_and_value_function<DTYPE, ENVIRONMENT, ActorCriticType::ACTOR_NETWORK_TYPE, ActorCriticType::CRITIC_NETWORK_TYPE>(actor_critic.actor, actor_critic.critic_1, std::string("full_training"), step_i);
//#endif
#ifdef LAYER_IN_C_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_EVALUATE_VISUALLY
//            for(int evaluation_i = 0; evaluation_i < 10; evaluation_i++){
//                ENVIRONMENT::State initial_state;
//                lic::sample_initial_state(env, initial_state, rng);
//                lic::evaluate_visual<ENVIRONMENT, UI, ActorCriticType::ACTOR_NETWORK_TYPE, ENVIRONMENT_STEP_LIMIT, 5>(env, ui, actor_critic.actor, initial_state);
//            }
#endif
        }
    }
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
        if(std::getenv("LAYER_IN_C_TEST_ENABLE_TIMING") != nullptr){
#ifdef LAYER_IN_C_TEST_MACHINE_LENOVO_P1
            ASSERT_LT(elapsed_seconds.count(), 6); // should be 5.5s when run in isolation
#endif
#ifdef LAYER_IN_C_TEST_MACHINE_MACBOOK_M1
            ASSERT_LT(elapsed_seconds.count(), 3); // should be 2.5s when run in isolation
#endif
        }
    }
    lic::free(ac_dev, critic_batch);
    lic::free(ac_dev, critic_training_buffers);
    lic::free(ac_dev, actor_batch);
    lic::free(ac_dev, actor_training_buffers);
    lic::free(ac_dev, off_policy_runner);
    lic::free(ac_dev, actor_critic);
}
