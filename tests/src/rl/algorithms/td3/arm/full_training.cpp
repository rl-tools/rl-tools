#include <layer_in_c/operations/arm.h>

namespace lic = layer_in_c;

#include <layer_in_c/nn/operations_generic.h>
using DEVICE = lic::devices::DefaultARM;

#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/operations_generic.h>

#include <layer_in_c/rl/utils/evaluation.h>
#ifndef LAYER_IN_C_DEPLOYMENT_ARDUINO
#include <chrono>
#include <iostream>
#endif

using DTYPE = float;

using PENDULUM_SPEC = lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>>;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters;

using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 16, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 16, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


using OPTIMIZER_PARAMETERS = typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;



constexpr DEVICE::index_t N_STEPS = 150000;
constexpr DEVICE::index_t EVALUATION_INTERVAL = 1000;
constexpr DEVICE::index_t N_EVALUATIONS = N_STEPS / EVALUATION_INTERVAL;
DTYPE evaluation_returns[N_EVALUATIONS];

constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 2000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>
>;
lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
ActorCriticType actor_critic;
const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);


void train(){
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;

    OPTIMIZER optimizer;

    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 1);

    bool ui = false;

    ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];

    lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::BuffersForwardBackward<ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];

    lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;

    lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
    lic::Matrix<lic::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;

    lic::malloc(device, actor_critic);
    lic::malloc(device, off_policy_runner);
    lic::malloc(device, critic_batch);
    lic::malloc(device, critic_training_buffers);
    lic::malloc(device, critic_buffers[0]);
    lic::malloc(device, critic_buffers[1]);
    lic::malloc(device, actor_batch);
    lic::malloc(device, actor_training_buffers);
    lic::malloc(device, actor_buffers_eval);
    lic::malloc(device, actor_buffers[0]);
    lic::malloc(device, actor_buffers[1]);
    lic::malloc(device, observations_mean);
    lic::malloc(device, observations_std);

    lic::init(device, actor_critic, optimizer, rng);
    lic::init(device, off_policy_runner, envs);
    lic::set_all(device, observations_mean, 0);
    lic::set_all(device, observations_std, 1);


#ifndef LAYER_IN_C_DEPLOYMENT_ARDUINO
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    for(int step_i = 0; step_i < N_STEPS; step_i+=OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        lic::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
        if(step_i % 100 == 0){
#ifdef LAYER_IN_C_DEPLOYMENT_ARDUINO
            Serial.printf("step: %d\n", step_i);
#else
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - start_time;
            std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
#endif

        }
        if(step_i > N_WARMUP_STEPS){

            for(int critic_i = 0; critic_i < 2; critic_i++){
                lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                lic::gather_batch(device, off_policy_runner, critic_batch, rng);
                lic::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

            if(step_i % 2 == 0){
                {
                    lic::gather_batch(device, off_policy_runner, actor_batch, rng);
                    lic::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                lic::update_critic_targets(device, actor_critic);
                lic::update_actor_target(device, actor_critic);
            }
        }
        if(step_i % EVALUATION_INTERVAL == 0){
            auto result = lic::evaluate(device, envs[0], ui, actor_critic.actor, lic::rl::utils::evaluation::Specification<10, ENVIRONMENT_STEP_LIMIT>(), observations_mean, observations_std, rng);
            if(N_EVALUATIONS > 0){
                evaluation_returns[(step_i / EVALUATION_INTERVAL) % N_EVALUATIONS] = result.mean;
            }
#ifdef LAYER_IN_C_DEPLOYMENT_ARDUINO
            Serial.printf("mean return: %f\n", result.mean);
#else
            std::cout << "Mean return: " << result.mean << std::endl;
#endif
        }
    }
#ifndef LAYER_IN_C_DEPLOYMENT_ARDUINO
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    }
#endif
    lic::free(device, actor_critic);
    lic::free(device, off_policy_runner);
    lic::free(device, critic_batch);
    lic::free(device, critic_training_buffers);
    lic::free(device, critic_buffers[0]);
    lic::free(device, critic_buffers[1]);
    lic::free(device, actor_batch);
    lic::free(device, actor_training_buffers);
    lic::free(device, actor_buffers_eval);
    lic::free(device, actor_buffers[0]);
    lic::free(device, actor_buffers[1]);
    lic::free(device, observations_mean);
    lic::free(device, observations_std);
}
