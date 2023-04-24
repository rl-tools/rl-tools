//#define BACKPROP_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
#define BACKPROP_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
#include <backprop_tools/operations/arm.h>

namespace bpt = backprop_tools;

#include <backprop_tools/nn/layers/dense/operations_arm/opt.h>
//#include <backprop_tools/nn/layers/dense/operations_arm/dsp.h>
#include <backprop_tools/nn/operations_generic.h>
using DEVICE = bpt::devices::arm::Generic<bpt::devices::DefaultARMSpecification>;

#include <backprop_tools/rl/environments/operations_generic.h>
#include <backprop_tools/nn_models/operations_generic.h>
#include <backprop_tools/rl/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>
#ifndef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
#include <chrono>
#include <iostream>
#endif

using DTYPE = float;
using CONTAINER_TYPE_TAG = bpt::MatrixDynamicTag;
using CONTAINER_TYPE_TAG_CRITIC = bpt::MatrixStaticTag;
using CONTAINER_TYPE_TAG_OFF_POLICY_RUNNER = bpt::MatrixStaticTag;
using CONTAINER_TYPE_TAG_TRAINING_BUFFERS = bpt::MatrixDynamicTag;

using PENDULUM_SPEC = bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>>;
typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
    constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
    constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3PendulumParameters;

using ActorStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE, CONTAINER_TYPE_TAG>;
using CriticStructureSpec = bpt::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE, CONTAINER_TYPE_TAG_CRITIC>;


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

using TD3_SPEC = bpt::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS, CONTAINER_TYPE_TAG>;
using ActorCriticType = bpt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;



constexpr DEVICE::index_t N_STEPS = 10000;
constexpr DEVICE::index_t EVALUATION_INTERVAL = 1000;
constexpr DEVICE::index_t N_EVALUATIONS = N_STEPS / EVALUATION_INTERVAL;
#ifndef LAYER_IN_C_DISABLE_EVALUATION
DTYPE evaluation_returns[N_EVALUATIONS];
#endif

constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 10000;
constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<
        DTYPE,
        DEVICE::index_t,
        ENVIRONMENT,
        1,
        REPLAY_BUFFER_CAP,
        ENVIRONMENT_STEP_LIMIT,
        bpt::rl::components::off_policy_runner::DefaultParameters<DTYPE>,
        false,
        0,
        CONTAINER_TYPE_TAG_OFF_POLICY_RUNNER
 >;
#ifdef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
EXTMEM bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
#else
bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC> off_policy_runner;
#endif
ActorCriticType actor_critic;

const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];

bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
bpt::rl::algorithms::td3::CriticTrainingBuffers<ActorCriticType::SPEC> critic_training_buffers;
CRITIC_NETWORK_TYPE::BuffersForwardBackward<ActorCriticType::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> critic_buffers;

bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<decltype(off_policy_runner)::SPEC, ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
bpt::rl::algorithms::td3::ActorTrainingBuffers<ActorCriticType::SPEC> actor_training_buffers;
ACTOR_NETWORK_TYPE::Buffers<ActorCriticType::SPEC::PARAMETERS::ACTOR_BATCH_SIZE, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> actor_buffers;
ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, CONTAINER_TYPE_TAG_TRAINING_BUFFERS> actor_buffers_eval;

typename CONTAINER_TYPE_TAG::template type<bpt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
typename CONTAINER_TYPE_TAG::template type<bpt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;


void train(){

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;

    OPTIMIZER optimizer;

    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 1);

    bool ui = false;

    bpt::malloc(device, actor_critic);
    bpt::malloc(device, off_policy_runner);
    bpt::malloc(device, critic_batch);
    bpt::malloc(device, critic_training_buffers);
    bpt::malloc(device, critic_buffers);
    bpt::malloc(device, actor_batch);
    bpt::malloc(device, actor_training_buffers);
    bpt::malloc(device, actor_buffers_eval);
    bpt::malloc(device, actor_buffers);
    bpt::malloc(device, observations_mean);
    bpt::malloc(device, observations_std);

#ifndef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
    std::cout << "malloc counter: " << device.malloc_counter << std::endl;
#endif
#endif


    bpt::init(device, actor_critic, optimizer, rng);
    bpt::init(device, off_policy_runner, envs);
    bpt::set_all(device, observations_mean, 0);
    bpt::set_all(device, observations_std, 1);


#ifndef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
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
        bpt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);
#ifdef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
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
                bpt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                bpt::gather_batch(device, off_policy_runner, critic_batch, rng);
                bpt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, optimizer, actor_buffers, critic_buffers, critic_training_buffers);
            }

            if(step_i % 2 == 0){
                {
                    bpt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    bpt::train_actor(device, actor_critic, actor_batch, optimizer, actor_buffers, critic_buffers, actor_training_buffers);
                }

                bpt::update_critic_targets(device, actor_critic);
                bpt::update_actor_target(device, actor_critic);
            }
        }
#ifndef LAYER_IN_C_DISABLE_EVALUATION
        if(step_i % EVALUATION_INTERVAL == 0){
            auto result = bpt::evaluate(device, envs[0], ui, actor_critic.actor, bpt::rl::utils::evaluation::Specification<10, ENVIRONMENT_STEP_LIMIT>(), observations_mean, observations_std, rng);
            if(N_EVALUATIONS > 0){
                evaluation_returns[(step_i / EVALUATION_INTERVAL) % N_EVALUATIONS] = result.mean;
            }
#ifdef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
            Serial.printf("mean return: %f\n", result.mean);
#else
            std::cout << "Mean return: " << result.mean << std::endl;
#endif
        }
#endif
    }
#ifndef BACKPROP_TOOLS_DEPLOYMENT_ARDUINO
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
    }
#endif
    bpt::free(device, actor_critic);
    bpt::free(device, off_policy_runner);
    bpt::free(device, critic_batch);
    bpt::free(device, critic_training_buffers);
    bpt::free(device, critic_buffers);
    bpt::free(device, actor_batch);
    bpt::free(device, actor_training_buffers);
    bpt::free(device, actor_buffers_eval);
    bpt::free(device, actor_buffers);
    bpt::free(device, observations_mean);
    bpt::free(device, observations_std);
}

