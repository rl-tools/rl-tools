// Group 1
#include <layer_in_c/operations/cuda/group_1.h>
#include <layer_in_c/operations/cpu/group_1.h>

// Group 2
#include <layer_in_c/operations/cuda/group_2.h>
#include <layer_in_c/operations/cpu/group_2.h>

// Group 3
#include <layer_in_c/operations/cuda/group_3.h>
#include <layer_in_c/operations/cpu/group_3.h>

#include <layer_in_c/nn/operations_cuda.h>
#include <layer_in_c/nn/loss_functions/mse/operations_cuda.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/nn_models/operations_cpu.h>

#include <layer_in_c/rl/components/replay_buffer/operations_cpu.h>
#include <layer_in_c/rl/components/replay_buffer/persist.h>
#include <layer_in_c/rl/components/off_policy_runner/operations_cpu.h>

#include <layer_in_c/rl/environments/pendulum/operations_cpu.h>

#include <layer_in_c/rl/components/off_policy_runner/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_cuda.h>
#include <layer_in_c/rl/algorithms/td3/operations_cpu.h>

#include "../components/replay_buffer.h"


#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace lic = layer_in_c;

class LAYER_IN_C_RL_CUDA : public ::testing::Test {
public:
    using DEVICE_CPU = lic::devices::DefaultCPU;
    using DEVICE_GPU = lic::devices::DefaultCUDA;
    using NN_DEVICE = DEVICE_CPU;
    using DTYPE = double;
    static constexpr DTYPE EPSILON = lic::utils::typing::is_same_v<DTYPE, float> ? 1e-5 : 1e-13;
    static constexpr DEVICE_CPU::index_t CAPACITY = 2000;
    static constexpr DEVICE_CPU::index_t BATCH_SIZE = 20;
//    using REPLAY_BUFFER_SPEC = lic::rl::components::replay_buffer::Specification<DTYPE, DEVICE_CPU::index_t, OBSERVATION_DIM, ACTION_DIM, CAPACITY>;
//    using REPLAY_BUFFER = lic::rl::components::ReplayBuffer<REPLAY_BUFFER_SPEC>;
    using PENDULUM_SPEC = lic::rl::environments::pendulum::Specification<DTYPE, DEVICE_CPU::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<PENDULUM_SPEC>;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<DTYPE, DEVICE_CPU::index_t, ENVIRONMENT, 1, CAPACITY, 100, lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>>;
    using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    using BATCH_SPEC = lic::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>;
    using BATCH_TYPE = lic::rl::components::off_policy_runner::Batch<BATCH_SPEC>;
    struct TD3_PARAMETERS: lic::rl::algorithms::td3::DefaultParameters<DTYPE, NN_DEVICE::index_t>{
        static constexpr typename NN_DEVICE::index_t ACTOR_BATCH_SIZE = BATCH_SIZE;
        static constexpr typename NN_DEVICE::index_t CRITIC_BATCH_SIZE = BATCH_SIZE;
    };
    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC , typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;
    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;
    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC, typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;
    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;
    using ACTOR_CRITIC_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = lic::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;
    using ACTOR_BUFFERS = lic::nn_models::mlp::NeuralNetworkBuffersForwardBackward<lic::nn_models::mlp::NeuralNetworkBuffersSpecification<ACTOR_NETWORK_SPEC, ACTOR_CRITIC_SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_cpu;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_cpu_2;
    OFF_POLICY_RUNNER_TYPE off_policy_runner_gpu_cpu;
    OFF_POLICY_RUNNER_TYPE* off_policy_runner_gpu_struct;
    BATCH_TYPE batch_cpu, batch_cpu_2;
    BATCH_TYPE batch_gpu;
    BATCH_TYPE* batch_gpu_struct;
    ACTOR_CRITIC_TYPE actor_critic_cpu, actor_critic_cpu_2;
    ACTOR_CRITIC_TYPE actor_critic_gpu;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_cpu;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_cpu_2;
    lic::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_gpu;
    ACTOR_BUFFERS actor_buffers_cpu;
    ACTOR_BUFFERS actor_buffers_cpu_2;
    ACTOR_BUFFERS actor_buffers_gpu;
protected:
    void SetUp() override {
        lic::init(device_gpu);
        auto rng_cpu = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
        auto rng_gpu = lic::random::default_engine(DEVICE_GPU::SPEC::RANDOM());
        // alloc
        lic::malloc(device_cpu, off_policy_runner_cpu);
        lic::malloc(device_cpu, off_policy_runner_cpu_2);
        lic::malloc(device_gpu, off_policy_runner_gpu_cpu);
        lic::malloc(device_cpu, batch_cpu);
        lic::malloc(device_cpu, batch_cpu_2);
        lic::malloc(device_gpu, batch_gpu);
        cudaMalloc(&off_policy_runner_gpu_struct, sizeof(OFF_POLICY_RUNNER_TYPE));
        cudaMalloc(&batch_gpu_struct, sizeof(BATCH_TYPE));
        lic::malloc(device_cpu, actor_critic_cpu);
        lic::malloc(device_cpu, actor_critic_cpu_2);
        lic::malloc(device_gpu, actor_critic_gpu);
        lic::malloc(device_cpu, critic_training_buffers_cpu);
        lic::malloc(device_cpu, critic_training_buffers_cpu_2);
        lic::malloc(device_gpu, critic_training_buffers_gpu);
        lic::malloc(device_cpu, actor_buffers_cpu);
        lic::malloc(device_cpu, actor_buffers_cpu_2);
        lic::malloc(device_gpu, actor_buffers_gpu);

        // init
        for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
            lic::test::rl::components::replay_buffer::sample(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], rng_cpu);
            lic::copy(device_gpu, device_cpu, off_policy_runner_gpu_cpu.replay_buffers[rb_i], off_policy_runner_cpu.replay_buffers[rb_i]);
        }
        lic::init(device_cpu, actor_critic_cpu, rng_cpu);

        // copy
        lic::check_status(device_gpu);
        cudaMemcpy(off_policy_runner_gpu_struct, &off_policy_runner_gpu_cpu, sizeof(OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
        lic::check_status(device_gpu);
        cudaMemcpy(batch_gpu_struct, &batch_gpu, sizeof(BATCH_TYPE), cudaMemcpyHostToDevice);
        lic::check_status(device_gpu);
        lic::copy(device_gpu, device_cpu, actor_critic_gpu, actor_critic_cpu);
    }

    void TearDown() override {
        lic::free(device_cpu, off_policy_runner_cpu);
        lic::free(device_cpu, off_policy_runner_cpu_2);
        lic::free(device_gpu, off_policy_runner_gpu_cpu);
        lic::free(device_cpu, batch_cpu);
        lic::free(device_cpu, batch_cpu_2);
        lic::free(device_gpu, batch_gpu);
        cudaFree(off_policy_runner_gpu_struct);
        cudaFree(batch_gpu_struct);
        lic::free(device_cpu, actor_critic_cpu);
        lic::free(device_cpu, actor_critic_cpu_2);
        lic::free(device_gpu, actor_critic_gpu);
        lic::free(device_cpu, critic_training_buffers_cpu);
        lic::free(device_cpu, critic_training_buffers_cpu_2);
        lic::free(device_gpu, critic_training_buffers_gpu);
        lic::free(device_cpu, actor_buffers_cpu);
        lic::free(device_cpu, actor_buffers_cpu_2);
        lic::free(device_gpu, actor_buffers_gpu);
    }
};


TEST_F(LAYER_IN_C_RL_CUDA, GATHER_BATCH) {

    auto rng_cpu = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = lic::random::default_engine(DEVICE_GPU::SPEC::RANDOM());
    for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
        lic::copy(device_cpu, device_gpu, off_policy_runner_cpu_2.replay_buffers[rb_i], off_policy_runner_gpu_cpu.replay_buffers[rb_i]);
        auto abs_diff = lic::abs_diff(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], off_policy_runner_cpu_2.replay_buffers[rb_i]);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }

    lic::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
    lic::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

    auto batch_observation_view = lic::view<DEVICE_CPU, typename decltype(off_policy_runner_cpu.replay_buffers[0].observations)::SPEC, BATCH_SIZE, BATCH_TYPE::OBSERVATION_DIM>(device_cpu, off_policy_runner_cpu.replay_buffers[0].observations, 0, 0);
    lic::print(device_cpu, batch_observation_view);
    std::cout << "BATCH" << std::endl;
    lic::print(device_cpu, batch_cpu.observations);
    std::cout << "BATCH GPU" << std::endl;
    lic::copy(device_cpu, device_gpu, batch_cpu_2, batch_gpu);
    lic::print(device_cpu, batch_cpu_2.observations);

    auto abs_diff_batch = lic::abs_diff(device_cpu, batch_cpu.observations, batch_cpu_2.observations);
    abs_diff_batch += lic::abs_diff(device_cpu, batch_cpu.actions, batch_cpu_2.actions);
    abs_diff_batch += lic::abs_diff(device_cpu, batch_cpu.next_observations, batch_cpu_2.next_observations);
    abs_diff_batch += lic::abs_diff(device_cpu, batch_cpu.rewards, batch_cpu_2.rewards);
    abs_diff_batch += lic::abs_diff(device_cpu, batch_cpu.terminated, batch_cpu_2.terminated);
    abs_diff_batch += lic::abs_diff(device_cpu, batch_cpu.truncated, batch_cpu_2.truncated);
    ASSERT_FLOAT_EQ(abs_diff_batch, 0);
}
TEST_F(LAYER_IN_C_RL_CUDA, TRAIN_CRITIC) {

    auto rng_cpu = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = lic::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    lic::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
    lic::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

    lic::target_action_noise(device_cpu, actor_critic_cpu, critic_training_buffers_cpu.target_next_action_noise, rng_cpu);
    lic::copy(device_gpu, device_cpu, critic_training_buffers_gpu.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
    lic::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
    auto abs_diff_target_next_action_noise = lic::abs_diff(device_cpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
    ASSERT_FLOAT_EQ(abs_diff_target_next_action_noise, 0);
//    lic::target_action_noise(device_gpu, actor_critic_gpu, critic_training_buffers_gpu.target_next_action_noise, rng_gpu);

    lic::zero_gradient(device_cpu, actor_critic_cpu.critic_1);
    lic::zero_gradient(device_gpu, actor_critic_gpu.critic_1);

    static_assert(BATCH_SPEC::BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);

    lic::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
    lic::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
    auto abs_diff_actor_critic = lic::abs_diff(device_cpu, actor_critic_cpu_2.actor_target, actor_critic_cpu.actor_target);
    ASSERT_FLOAT_EQ(abs_diff_actor_critic, 0);

    lic::evaluate(device_cpu, actor_critic_cpu.actor.input_layer, batch_cpu.observations, actor_buffers_cpu.tick);
    lic::evaluate(device_gpu, actor_critic_gpu.actor.input_layer, batch_gpu.observations, actor_buffers_gpu.tick);
    lic::copy(device_cpu, device_gpu, actor_buffers_cpu_2, actor_buffers_gpu);
    auto abs_diff_tick = lic::abs_diff(device_cpu, actor_buffers_cpu_2.tick, actor_buffers_cpu.tick);
    ASSERT_LT(abs_diff_tick, EPSILON);

    lic::evaluate(device_cpu, actor_critic_cpu.actor_target, batch_cpu.next_observations, critic_training_buffers_cpu.next_actions, actor_buffers_cpu);
    lic::evaluate(device_gpu, actor_critic_gpu.actor_target, batch_gpu.next_observations, critic_training_buffers_gpu.next_actions, actor_buffers_gpu);

    lic::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
    auto abs_diff_next_actions = lic::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_actions, critic_training_buffers_cpu.next_actions);
    ASSERT_LT(abs_diff_next_actions, EPSILON);

    lic::noisy_next_actions(device_cpu, critic_training_buffers_cpu);
    lic::noisy_next_actions(device_gpu, critic_training_buffers_gpu);
    lic::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
    lic::check_status(device_gpu);
    lic::print(device_cpu, critic_training_buffers_cpu_2.next_actions);

}

TEST_F(LAYER_IN_C_RL_CUDA, VIEW_COPY_PROBLEM) {

    auto rng_cpu = lic::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = lic::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    lic::randn(device_cpu, batch_cpu.next_observations, rng_cpu);
    lic::copy_structure_mismatch(device_gpu, device_cpu, batch_gpu.next_observations, batch_cpu.next_observations);
    lic::copy_structure_mismatch(device_cpu, device_gpu, batch_cpu_2.next_observations, batch_gpu.next_observations);

    auto abs_diff_next_observations = lic::abs_diff(device_cpu, batch_cpu_2.next_observations, batch_cpu.next_observations);
    ASSERT_LT(abs_diff_next_observations, EPSILON);
}
