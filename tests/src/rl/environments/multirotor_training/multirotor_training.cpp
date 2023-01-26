//#define USE_PENDULUM
#include <layer_in_c/operations/cpu_mkl.h>
#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/multirotor/parameters/default.h>
#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/nn/operations_cpu_mkl.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/environments/multirotor/operations_cpu.h>
#ifdef USE_PENDULUM
#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>
#endif
#include <layer_in_c/rl/components/off_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>

#include <layer_in_c/rl/environments/multirotor/ui.h>

#include <layer_in_c/rl/utils/evaluation.h>

#include "parameters.h"

#include <gtest/gtest.h>
#include <ctime>
#include <iostream>
#include <filesystem>




namespace lic = layer_in_c;
using DTYPE = float;


//using DEVICE = lic::devices::DefaultCPU_MKL;
using DEVICE = lic::devices::CPU_MKL<lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>>;


#ifndef USE_PENDULUM
auto parameters = parameters_0::parameters<DTYPE, DEVICE::index_t>;
using PARAMETERS = decltype(parameters);
using REWARD_FUNCTION = PARAMETERS::MDP::REWARD_FUNCTION;
typedef lic::rl::environments::multirotor::Specification<DTYPE, DEVICE::index_t, PARAMETERS, lic::rl::environments::multirotor::StaticParameters> ENVIRONMENT_SPEC;
typedef lic::rl::environments::Multirotor<ENVIRONMENT_SPEC> ENVIRONMENT;
#endif


#ifdef USE_PENDULUM
typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
#endif


template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, DEVICE::index_t>{
    static constexpr typename DEVICE::index_t CRITIC_BATCH_SIZE = 256;
    static constexpr typename DEVICE::index_t CRITIC_TRAINING_INTERVAL = 10;
    static constexpr typename DEVICE::index_t ACTOR_BATCH_SIZE = 256;
    static constexpr typename DEVICE::index_t ACTOR_TRAINING_INTERVAL = 20;
    static constexpr typename DEVICE::index_t CRITIC_TARGET_UPDATE_INTERVAL = 20;
    static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.25;
    static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
};

using TD3_PARAMETERS = TD3PendulumParameters<DTYPE>;

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 256, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;

using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 1000;
using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
>;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = 30000;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING) {

    time_t now;
    time(&now);
    char buf[sizeof "2011-10-08T07:07:09Z"];
    strftime(buf, sizeof buf, "%FT%TZ", gmtime(&now));

    std::string logs_dir = "logs";
    if (!std::filesystem::is_directory(logs_dir.c_str()) || !std::filesystem::exists(logs_dir.c_str())) {
        std::filesystem::create_directory(logs_dir.c_str());
    }
    std::string log_dir = logs_dir + "/" + std::string(buf);
    if (!std::filesystem::is_directory(log_dir.c_str()) || !std::filesystem::exists(log_dir.c_str())) {
        std::filesystem::create_directory(log_dir.c_str());
    }

    std::string log_file = log_dir + "/" + std::string("data.tfevents");
    std::cout << "Logging to " << log_file << std::endl;
    TensorBoardLogger tb_logger(log_file.c_str());
    DEVICE::SPEC::LOGGING logger;
    logger.tb = &tb_logger;
    DEVICE device(logger);

#ifndef USE_PENDULUM
    lic::rl::environments::multirotor::UI<ENVIRONMENT> ui;
    ui.host = "localhost";
    ui.port = "8080";
    lic::init(device, ui);
#endif

    std::mt19937 rng(3);
    lic::malloc(device, actor_critic);
    lic::init(device, actor_critic, rng);
#ifndef USE_PENDULUM
//    parameters.mdp.init = lic::rl::environments::multirotor::parameters::init::simple<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
    parameters.mdp.init = lic::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
    ENVIRONMENT env({parameters});
#else
    ENVIRONMENT env;
#endif
    lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner = {env};

    lic::malloc(device, off_policy_runner);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    lic::malloc(device, critic_batch);
    lic::malloc(device, critic_training_buffers);

    lic::rl::components::replay_buffer::Batch<decltype(off_policy_runner.replay_buffer)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    lic::malloc(device, actor_batch);
    lic::malloc(device, actor_training_buffers);

    for(int step_i = 0; step_i < 1000000; step_i++){
        device.logger.step = step_i;
        if(step_i > REPLAY_BUFFER_CAP){
            std::cout << "warning: replay buffer is rolling over" << std::endl;
        }
        lic::step(device, off_policy_runner, actor_critic.actor, rng);
#ifndef USE_PENDULUM
//        lic::set_state(device, ui, off_policy_runner.state);
//        std::this_thread::sleep_for(std::chrono::milliseconds((int)(parameters.integration.dt * 1000)));
#endif
        if(step_i % 1000 == 0){
            std::cout << "step_i: " << step_i << std::endl;
        }
        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % ActorCriticType::SPEC::PARAMETERS::CRITIC_TRAINING_INTERVAL == 0){
                for(int critic_i = 0; critic_i < 2; critic_i++){
                    lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                    lic::gather_batch(device, off_policy_runner.replay_buffer, critic_batch, rng);
                    DTYPE critic_loss = lic::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, critic_training_buffers);
                    if(critic_i == 0){
                        lic::logging::add_scalar(device.logger, "critic_1_loss", critic_loss, 100);
                    }
                }
                lic::update_critic_targets(device, actor_critic);
            }
            if(step_i % ActorCriticType::SPEC::PARAMETERS::ACTOR_TRAINING_INTERVAL == 0){
                lic::gather_batch(device, off_policy_runner.replay_buffer, actor_batch, rng);
                DTYPE actor_value = lic::train_actor(device, actor_critic, actor_batch, actor_training_buffers);
                lic::logging::add_scalar(device.logger, "actor_value", actor_value, 100);
                lic::update_actor_target(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(device, env, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
        }
    }
    lic::free(device, critic_batch);
    lic::free(device, critic_training_buffers);
    lic::free(device, actor_batch);
    lic::free(device, actor_training_buffers);
}
