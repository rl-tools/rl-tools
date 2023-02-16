// ------------ Groups 1 ------------
#include <layer_in_c/operations/cuda/group_1.h>
#include <layer_in_c/operations/cpu_tensorboard/group_1.h>
// ------------ Groups 2 ------------
#include <layer_in_c/operations/cuda/group_2.h>
#include <layer_in_c/operations/cpu_tensorboard/group_2.h>
// ------------ Groups 3 ------------
#include <layer_in_c/operations/cuda/group_3.h>
#include <layer_in_c/operations/cpu_tensorboard/group_3.h>

namespace lic = layer_in_c;

#include <layer_in_c/nn/operations_cuda.h>
using DEV_SPEC_INIT = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, lic::devices::logging::CPU_TENSORBOARD>;
using DEVICE_INIT = lic::devices::CPU<DEV_SPEC_INIT>;
using DEVICE = lic::devices::DefaultCUDA;
using DEV_SPEC = DEVICE::SPEC;

#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_generic.h>


#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <filesystem>

using DTYPE = float;

typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

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
using ACTOR_CRITIC_TYPE = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = 500000;
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
using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
using CRITIC_BATCH_TYPE = lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
using ACTOR_BATCH_TYPE = lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;

const DTYPE STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TEST(LAYER_IN_C_RL_CUDA_TD3, TEST_FULL_TRAINING) {
    DEVICE_INIT::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_INIT device_init;
    ACTOR_CRITIC_TYPE actor_critic_init;
    ACTOR_CRITIC_TYPE actor_critic;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_init, off_policy_runner;
    OFF_POLICY_RUNNER_TYPE* off_policy_runner_pointer;

    CRITIC_BATCH_TYPE critic_batch;
    CRITIC_BATCH_TYPE* critic_batch_pointer;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_TYPE ::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::BuffersForwardBackward<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];

    ACTOR_BATCH_TYPE actor_batch;
    ACTOR_BATCH_TYPE* actor_batch_pointer;
    lic::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::Buffers<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    ACTOR_NETWORK_TYPE::Buffers<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval_init;

    lic::init(device);
    device_init.logger = &logger;
    lic::construct(device_init, device_init.logger);
    auto rng_init = lic::random::default_engine(DEVICE_INIT::SPEC::RANDOM());
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM());
    ENVIRONMENT envs[decltype(off_policy_runner_init)::N_ENVIRONMENTS];
    bool ui = false;
    
    
    lic::malloc(device_init, actor_critic_init);
    lic::malloc(device, actor_critic);
    lic::malloc(device_init, off_policy_runner_init);
    lic::malloc(device, off_policy_runner);
    cudaMalloc(&off_policy_runner_pointer, sizeof(OFF_POLICY_RUNNER_TYPE));
    lic::check_status(device);

    lic::malloc(device, critic_batch);
    cudaMalloc(&critic_batch_pointer, sizeof(CRITIC_BATCH_TYPE));
    lic::check_status(device);
    lic::malloc(device, critic_training_buffers);
    lic::malloc(device, critic_buffers[0]);
    lic::malloc(device, critic_buffers[1]);

    lic::malloc(device, actor_batch);
    cudaMalloc(&actor_batch_pointer, sizeof(ACTOR_BATCH_TYPE));
    lic::check_status(device);
    lic::malloc(device, actor_training_buffers);
    lic::malloc(device, actor_buffers_eval);
    lic::malloc(device_init, actor_buffers_eval_init);
    lic::malloc(device, actor_buffers[0]);
    lic::malloc(device, actor_buffers[1]);

    lic::init(device_init, actor_critic_init, rng_init);
    lic::copy(device, device_init, actor_critic, actor_critic_init);
    lic::init(device_init, off_policy_runner_init, envs);
    cudaMemcpy(off_policy_runner_pointer, &off_policy_runner, sizeof(OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);
    cudaMemcpy(actor_batch_pointer, &actor_batch, sizeof(ACTOR_BATCH_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);
    cudaMemcpy(critic_batch_pointer, &critic_batch, sizeof(CRITIC_BATCH_TYPE), cudaMemcpyHostToDevice);
    lic::check_status(device);

    auto start_time = std::chrono::high_resolution_clock::now();

    constexpr DEVICE::index_t step_limit = 15000;
    for(int step_i = 0; step_i < step_limit; step_i++){
        lic::step(device_init, off_policy_runner_init, actor_critic_init.actor, actor_buffers_eval_init, rng_init);

        if(off_policy_runner_init.step > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - start_time;
                std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
            }

            lic::copy(device, device_init, off_policy_runner.replay_buffers[0], off_policy_runner_init.replay_buffers[0]);
            cudaMemcpy(off_policy_runner_pointer, &off_policy_runner, sizeof(OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
            lic::check_status(device);

            for(int critic_i = 0; critic_i < 2; critic_i++){
                lic::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                lic::gather_batch(device, off_policy_runner_pointer, critic_batch_pointer, rng);
                lic::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

            if(step_i % 2 == 0){
                {
                    lic::gather_batch(device, off_policy_runner_pointer, actor_batch_pointer, rng);
                    lic::train_actor(device, actor_critic, actor_batch, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                lic::update_critic_targets(device, actor_critic);
                lic::update_actor_target(device, actor_critic);
            }
        }
        if(step_i % 1000 == 0){
            lic::copy(device_init, device, actor_critic_init, actor_critic);
            DTYPE mean_return = lic::evaluate<DEVICE_INIT, ENVIRONMENT, decltype(ui), decltype(actor_critic_init.actor), decltype(rng_init), ENVIRONMENT_STEP_LIMIT, true>(device_init, envs[0], ui, actor_critic_init.actor, 1, rng_init);
            std::cout << "Mean return: " << mean_return << std::endl;
        }
    }
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl; // 90s, 15x of CPU BLAS => todo: investigate individual kernel timings
    }
    lic::free(device, critic_batch);
    lic::free(device, critic_training_buffers);
    lic::free(device, actor_batch);
    lic::free(device, actor_training_buffers);
    lic::free(device, off_policy_runner);
    lic::free(device, actor_critic);
}
