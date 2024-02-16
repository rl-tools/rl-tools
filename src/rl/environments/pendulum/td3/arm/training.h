//#define RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
#define RL_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
#include <rl_tools/operations/arm.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
//#include <rl_tools/nn/layers/dense/operations_arm/dsp.h>
#include <rl_tools/nn/operations_generic.h>
using DEVICE = rlt::devices::arm::Generic<rlt::devices::DefaultARMSpecification>;

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/operations_generic.h>

#include <rl_tools/rl/utils/evaluation.h>
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
#include <chrono>
#include <iostream>
#endif

using DTYPE = float;
using CONTAINER_TYPE_TAG = rlt::MatrixDynamicTag;
using CONTAINER_TYPE_TAG_CRITIC = rlt::MatrixStaticTag;
using CONTAINER_TYPE_TAG_OFF_POLICY_RUNNER = rlt::MatrixStaticTag;
using CONTAINER_TYPE_TAG_TRAINING_BUFFERS = rlt::MatrixDynamicTag;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, rlt::rl::environments::pendulum::DefaultParameters<DTYPE>>;
typedef rlt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

struct TD3PendulumParameters: rlt::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters;

using ActorStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE, CONTAINER_TYPE_TAG>;
using CriticStructureSpec = rlt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE, CONTAINER_TYPE_TAG_CRITIC>;


using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<DTYPE, typename DEVICE::index_t>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
using ACTOR_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
using ACTOR_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

using ACTOR_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
using ACTOR_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

using CRITIC_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
using CRITIC_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

using CRITIC_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
using CRITIC_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

using TD3_SPEC = rlt::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS, CONTAINER_TYPE_TAG>;
using ActorCriticType = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;



constexpr DEVICE::index_t N_STEPS = 10000;
constexpr DEVICE::index_t EVALUATION_INTERVAL = 1000;
constexpr DEVICE::index_t N_EVALUATIONS = N_STEPS / EVALUATION_INTERVAL;
#ifndef RL_TOOLS_DISABLE_EVALUATION
DTYPE evaluation_returns[N_EVALUATIONS];
#endif

constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 10000;
constexpr typename DEVICE::index_t EPISODE_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        1,
        false,
        REPLAY_BUFFER_CAP,
        EPISODE_STEP_LIMIT,
        false,
        false,
        0,
        CONTAINER_TYPE_TAG_OFF_POLICY_RUNNER
 >;
#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
EXTMEM rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
#else
rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
#endif
ActorCriticType actor_critic;

const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];

rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
rlt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
CRITIC_NETWORK_TYPE::Buffer<ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> critic_buffers;

rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
rlt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
ACTOR_NETWORK_TYPE::Buffer<ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> actor_buffers;
ACTOR_NETWORK_TYPE::Buffer<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> actor_buffers_eval;

typename CONTAINER_TYPE_TAG::template type<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
typename CONTAINER_TYPE_TAG::template type<rlt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;


void train(){

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = logger;

    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 1);

    rlt::rl::environments::DummyUI ui;

    rlt::malloc(device, actor_critic);
    rlt::malloc(device, off_policy_runner);
    rlt::malloc(device, critic_batch);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device, critic_buffers);
    rlt::malloc(device, actor_batch);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device, actor_buffers_eval);
    rlt::malloc(device, actor_buffers);
    rlt::malloc(device, observations_mean);
    rlt::malloc(device, observations_std);

#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
#ifdef RL_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
    std::cout << "malloc counter: " << device.malloc_counter << std::endl;
#endif
#endif


    rlt::init(device, actor_critic, rng);
    rlt::init(device, off_policy_runner, envs);
    rlt::set_all(device, observations_mean, 0);
    rlt::set_all(device, observations_std, 1);


#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
    auto start_time = std::chrono::high_resolution_clock::now();
    std::cout << "ActorCritic size: " << sizeof(actor_critic) << std::endl;
    std::cout << "ActorCritic.actor size: " << sizeof(actor_critic.actor) << std::endl;
    std::cout << "ActorCritic.actor_target size: " << sizeof(actor_critic.actor_target) << std::endl;
    std::cout << "ActorCritic.critic_1 size: " << sizeof(actor_critic.critic_1) << std::endl;
    std::cout << "ActorCritic.critic_2 size: " << sizeof(actor_critic.critic_2) << std::endl;
    std::cout << "ActorCritic.critic_target_1 size: " << sizeof(actor_critic.critic_target_1) << std::endl;
    std::cout << "ActorCritic.critic_target_2 size: " << sizeof(actor_critic.critic_target_2) << std::endl;
    std::cout << "OffPolicyRunner size: " << sizeof(off_policy_runner) << std::endl;
    std::cout << "OffPolicyRunner.replay_buffers size: " << sizeof(off_policy_runner.replay_buffers) << std::endl;
    std::cout << "CriticBatch size: " << sizeof(critic_batch) << std::endl;
    std::cout << "CriticTrainingBuffers size: " << sizeof(critic_training_buffers) << std::endl;
    std::cout << "CriticBuffers size: " << sizeof(critic_buffers) << std::endl;
    std::cout << "ActorBatch size: " << sizeof(actor_batch) << std::endl;
    std::cout << "ActorTrainingBuffers size: " << sizeof(actor_training_buffers) << std::endl;
    std::cout << "ActorBuffers size: " << sizeof(actor_buffers) << std::endl;
    std::cout << "Total: " << sizeof(actor_critic) + sizeof(off_policy_runner) + sizeof(critic_batch) + sizeof(critic_training_buffers) + sizeof(critic_buffers) + sizeof(actor_batch) + sizeof(actor_training_buffers) + sizeof(actor_buffers) << std::endl;
#endif

    for(int step_i = 0; step_i < N_STEPS; step_i+=OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        rlt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
        if(step_i % 100 == 0){
            Serial.printf("step: %d\n", step_i);
#else
        if(step_i % 1000 == 0){
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed_seconds = current_time - start_time;
            std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
#endif

        }
        if(step_i > N_WARMUP_STEPS){

            for(int critic_i = 0; critic_i < 2; critic_i++){
                rlt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                rlt::gather_batch(device, off_policy_runner, critic_batch, rng);
                rlt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_critic.critic_optimizers[critic_i], actor_buffers, critic_buffers, critic_training_buffers);
            }

            if(step_i % 2 == 0){
                {
                    rlt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    rlt::train_actor(device, actor_critic, actor_batch, actor_critic.actor_optimizer, actor_buffers, critic_buffers, actor_training_buffers);
                }

                rlt::update_critic_targets(device, actor_critic);
                rlt::update_actor_target(device, actor_critic);
            }
        }
#ifndef RL_TOOLS_DISABLE_EVALUATION
        if(step_i % EVALUATION_INTERVAL == 0){
            auto result = rlt::evaluate(device, envs[0], ui, actor_critic.actor, rlt::rl::utils::evaluation::Specification<10, EPISODE_STEP_LIMIT>(), observations_mean, observations_std, actor_buffers, rng);
            if(N_EVALUATIONS > 0){
                evaluation_returns[(step_i / EVALUATION_INTERVAL) % N_EVALUATIONS] = result.returns_mean;
            }
#ifdef RL_TOOLS_DEPLOYMENT_ARDUINO
            Serial.printf("mean return: %f\n", result.returns_mean);
#else
            std::cout << "Mean return: " << result.returns_mean << std::endl;
#endif
        }
#endif
    }
#ifndef RL_TOOLS_DEPLOYMENT_ARDUINO
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    }
#endif
    rlt::free(device, actor_critic);
    rlt::free(device, off_policy_runner);
    rlt::free(device, critic_batch);
    rlt::free(device, critic_training_buffers);
    rlt::free(device, critic_buffers);
    rlt::free(device, actor_batch);
    rlt::free(device, actor_training_buffers);
    rlt::free(device, actor_buffers_eval);
    rlt::free(device, actor_buffers);
    rlt::free(device, observations_mean);
    rlt::free(device, observations_std);
}

