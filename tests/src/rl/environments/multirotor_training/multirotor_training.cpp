#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/rl/environments/multirotor/operations_cpu.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_generic.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>


#include <layer_in_c/rl/utils/evaluation.h>

#include <tensorboard_logger.h>
#include <gtest/gtest.h>
#include <ctime>
#include <iostream>
#include <filesystem>


namespace lic = layer_in_c;
using DTYPE = float;

using DEVICE = lic::devices::DefaultCPU;
typedef lic::rl::environments::multirotor::Specification<DTYPE, lic::rl::environments::multirotor::StaticParameters> ENVIRONMENT_SPEC;
typedef lic::rl::environments::Multirotor<DEVICE, ENVIRONMENT_SPEC> ENVIRONMENT;
auto parameters = lic::rl::environments::multirotor::default_parameters<DTYPE, DEVICE::index_t(4)>;


using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY>;

using AC_DEVICE = lic::devices::DefaultCPU;
template <typename T>
struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, AC_DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using NN_DEVICE = lic::devices::DefaultCPU;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3PendulumParameters<DTYPE>>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;


constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
AC_DEVICE::SPEC::LOGGING logger;
AC_DEVICE device(logger);
NN_DEVICE nn_device(logger);
lic::rl::components::OffPolicyRunner<
        lic::rl::components::off_policy_runner::Specification<
                DTYPE,
                AC_DEVICE::index_t,
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

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, TEST_FULL_TRAINING) {
    TensorBoardLoggerOptions tb_opts;
    tb_opts.flush_period_s(1);

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
    TensorBoardLogger tb_logger(log_file.c_str());
    std::mt19937 rng(2);
    lic::malloc(device, actor_critic);
    lic::init(device, actor_critic, rng);
    parameters.init = lic::rl::environments::multirotor::simple_init_parameters<DTYPE, DEVICE::index_t(4)>;
    ENVIRONMENT env({parameters});
    off_policy_runner.env = env;

    for(int step_i = 0; step_i < 1000000; step_i++){
        if(step_i > REPLAY_BUFFER_CAP){
            std::cout << "warning: replay buffer is rolling over" << std::endl;
        }
        lic::step(device, off_policy_runner, actor_critic.actor, rng);
        if(off_policy_runner.replay_buffer.full || off_policy_runner.replay_buffer.position > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                std::cout << "step_i: " << step_i << std::endl;
            }
            DTYPE critic_1_loss = lic::train_critic(device, actor_critic, actor_critic.critic_1, off_policy_runner.replay_buffer, rng);
            lic::train_critic(device, actor_critic, actor_critic.critic_2, off_policy_runner.replay_buffer, rng);
            tb_logger.add_scalar("critic_1_loss", step_i, critic_1_loss);
//            std::cout << "Critic 1 loss: " << critic_1_loss << std::endl;
            if(step_i % 2 == 0){
                lic::train_actor(device, actor_critic, off_policy_runner.replay_buffer, rng);
                lic::update_targets(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            DTYPE mean_return = lic::evaluate<DEVICE, ENVIRONMENT, decltype(actor_critic.actor), typeof(rng), ENVIRONMENT_STEP_LIMIT, true>(device, env, actor_critic.actor, 1, rng);
            std::cout << "Mean return: " << mean_return << std::endl;
        }
    }
}
