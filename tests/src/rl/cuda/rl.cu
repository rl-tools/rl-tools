// Group 1
#include <backprop_tools/operations/cuda/group_1.h>
#include <backprop_tools/operations/cpu/group_1.h>
#include <backprop_tools/operations/cpu_mkl/group_1.h>

// Group 2
#include <backprop_tools/operations/cuda/group_2.h>
#include <backprop_tools/operations/cpu/group_2.h>
#include <backprop_tools/operations/cpu_mkl/group_2.h>

// Group 3
#include <backprop_tools/operations/cuda/group_3.h>
#include <backprop_tools/operations/cpu/group_3.h>
#include <backprop_tools/operations/cpu_mkl/group_3.h>

#include <backprop_tools/nn/operations_cpu_mkl.h>
#include <backprop_tools/nn/operations_cuda.h>
#include <backprop_tools/nn/loss_functions/mse/operations_cuda.h>
#include <backprop_tools/nn_models/operations_generic.h>

#include <backprop_tools/rl/components/replay_buffer/operations_cpu.h>
#include <backprop_tools/rl/components/replay_buffer/persist.h>
#include <backprop_tools/rl/components/off_policy_runner/operations_cpu.h>

#include <backprop_tools/rl/environments/pendulum/operations_cpu.h>

#include <backprop_tools/rl/components/off_policy_runner/operations_cuda.h>
#include <backprop_tools/rl/algorithms/td3/operations_cuda.h>
#include <backprop_tools/rl/algorithms/td3/operations_cpu.h>

#include "../components/replay_buffer.h"


#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace bpt = backprop_tools;

class BACKPROP_TOOLS_RL_CUDA : public ::testing::Test {
public:
    using DEVICE_CPU = bpt::devices::DefaultCPU_MKL;
    using DEVICE_GPU = bpt::devices::DefaultCUDA;
    using NN_DEVICE = DEVICE_CPU;
    using DTYPE = double;
    static constexpr DEVICE_CPU::index_t CAPACITY = 20000;
    static constexpr DEVICE_CPU::index_t BATCH_SIZE = 256;
    static constexpr DTYPE EPSILON = (bpt::utils::typing::is_same_v<DTYPE, float> ? 1e-5 : 1e-10);// * BATCH_SIZE;
//    using REPLAY_BUFFER_SPEC = bpt::rl::components::replay_buffer::Specification<DTYPE, DEVICE_CPU::index_t, OBSERVATION_DIM, ACTION_DIM, CAPACITY>;
//    using REPLAY_BUFFER = bpt::rl::components::ReplayBuffer<REPLAY_BUFFER_SPEC>;
    using PENDULUM_SPEC = bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE_CPU::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>>;
    using ENVIRONMENT = bpt::rl::environments::Pendulum<PENDULUM_SPEC>;
    using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<DTYPE, DEVICE_CPU::index_t, ENVIRONMENT, 1, CAPACITY, 100, bpt::rl::components::off_policy_runner::DefaultParameters<DTYPE>>;
    using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    using BATCH_SPEC = bpt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, BATCH_SIZE>;
    using BATCH_TYPE = bpt::rl::components::off_policy_runner::Batch<BATCH_SPEC>;
    struct TD3_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<DTYPE, NN_DEVICE::index_t>{
        static constexpr typename NN_DEVICE::index_t ACTOR_BATCH_SIZE = BATCH_SIZE;
        static constexpr typename NN_DEVICE::index_t CRITIC_BATCH_SIZE = BATCH_SIZE;
    };
    using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
    using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<DTYPE, typename DEVICE_GPU::index_t>;
    using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;
    using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;
    using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;
    using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;
    using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<DTYPE, NN_DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;
    using ACTOR_BUFFERS = bpt::nn_models::mlp::NeuralNetworkBuffers<bpt::nn_models::mlp::NeuralNetworkBuffersSpecification<ACTOR_NETWORK_SPEC, ACTOR_CRITIC_SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
    using CRITIC_BUFFERS = bpt::nn_models::mlp::NeuralNetworkBuffers<bpt::nn_models::mlp::NeuralNetworkBuffersSpecification<CRITIC_NETWORK_SPEC, ACTOR_CRITIC_SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
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
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_cpu;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_cpu_2;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_SPEC> critic_training_buffers_gpu;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_SPEC> actor_training_buffers_cpu;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_SPEC> actor_training_buffers_cpu_2;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_SPEC> actor_training_buffers_gpu;
    ACTOR_BUFFERS actor_buffers_cpu;
    ACTOR_BUFFERS actor_buffers_cpu_2;
    ACTOR_BUFFERS actor_buffers_gpu;
    CRITIC_BUFFERS critic_buffers_cpu;
    CRITIC_BUFFERS critic_buffers_cpu_2;
    CRITIC_BUFFERS critic_buffers_gpu;
    using TI = typename DEVICE_GPU::index_t;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, CRITIC_NETWORK_TYPE::SPEC::BATCH_SIZE, 1>> d_critic_output_cpu;
    bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, TI, CRITIC_NETWORK_TYPE::SPEC::BATCH_SIZE, 1>> d_critic_output_gpu;
protected:
    void SetUp() override {
        bpt::init(device_gpu);
        auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
        auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());
        // alloc
        bpt::malloc(device_cpu, off_policy_runner_cpu);
        bpt::malloc(device_cpu, off_policy_runner_cpu_2);
        bpt::malloc(device_gpu, off_policy_runner_gpu_cpu);
        bpt::malloc(device_cpu, batch_cpu);
        bpt::malloc(device_cpu, batch_cpu_2);
        bpt::malloc(device_gpu, batch_gpu);
        cudaMalloc(&off_policy_runner_gpu_struct, sizeof(OFF_POLICY_RUNNER_TYPE));
        cudaMalloc(&batch_gpu_struct, sizeof(BATCH_TYPE));
        bpt::malloc(device_cpu, actor_critic_cpu);
        bpt::malloc(device_cpu, actor_critic_cpu_2);
        bpt::malloc(device_gpu, actor_critic_gpu);
        bpt::malloc(device_cpu, critic_training_buffers_cpu);
        bpt::malloc(device_cpu, critic_training_buffers_cpu_2);
        bpt::malloc(device_gpu, critic_training_buffers_gpu);
        bpt::malloc(device_cpu, actor_buffers_cpu);
        bpt::malloc(device_cpu, actor_buffers_cpu_2);
        bpt::malloc(device_gpu, actor_buffers_gpu);
        bpt::malloc(device_cpu, critic_buffers_cpu);
        bpt::malloc(device_cpu, critic_buffers_cpu_2);
        bpt::malloc(device_gpu, critic_buffers_gpu);
        bpt::malloc(device_cpu, actor_training_buffers_cpu);
        bpt::malloc(device_cpu, actor_training_buffers_cpu_2);
        bpt::malloc(device_gpu, actor_training_buffers_gpu);
        bpt::malloc(device_cpu, d_critic_output_cpu);
        bpt::malloc(device_gpu, d_critic_output_gpu);

        // init
        for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
            bpt::test::rl::components::replay_buffer::sample(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], rng_cpu);
            bpt::copy(device_gpu, device_cpu, off_policy_runner_gpu_cpu.replay_buffers[rb_i], off_policy_runner_cpu.replay_buffers[rb_i]);
        }
        bpt::init(device_cpu, actor_critic_cpu, rng_cpu);

        // copy
        bpt::check_status(device_gpu);
        cudaMemcpy(off_policy_runner_gpu_struct, &off_policy_runner_gpu_cpu, sizeof(OFF_POLICY_RUNNER_TYPE), cudaMemcpyHostToDevice);
        bpt::check_status(device_gpu);
        cudaMemcpy(batch_gpu_struct, &batch_gpu, sizeof(BATCH_TYPE), cudaMemcpyHostToDevice);
        bpt::check_status(device_gpu);
        bpt::copy(device_gpu, device_cpu, actor_critic_gpu, actor_critic_cpu);
    }

    void TearDown() override {
        bpt::free(device_cpu, off_policy_runner_cpu);
        bpt::free(device_cpu, off_policy_runner_cpu_2);
        bpt::free(device_gpu, off_policy_runner_gpu_cpu);
        bpt::free(device_cpu, batch_cpu);
        bpt::free(device_cpu, batch_cpu_2);
        bpt::free(device_gpu, batch_gpu);
        cudaFree(off_policy_runner_gpu_struct);
        cudaFree(batch_gpu_struct);
        bpt::free(device_cpu, actor_critic_cpu);
        bpt::free(device_cpu, actor_critic_cpu_2);
        bpt::free(device_gpu, actor_critic_gpu);
        bpt::free(device_cpu, critic_training_buffers_cpu);
        bpt::free(device_cpu, critic_training_buffers_cpu_2);
        bpt::free(device_gpu, critic_training_buffers_gpu);
        bpt::free(device_cpu, actor_buffers_cpu);
        bpt::free(device_cpu, actor_buffers_cpu_2);
        bpt::free(device_gpu, actor_buffers_gpu);
        bpt::free(device_cpu, critic_buffers_cpu);
        bpt::free(device_cpu, critic_buffers_cpu_2);
        bpt::free(device_gpu, critic_buffers_gpu);
        bpt::free(device_cpu, actor_training_buffers_cpu);
        bpt::free(device_cpu, actor_training_buffers_cpu_2);
        bpt::free(device_gpu, actor_training_buffers_gpu);
    }
};

TEST_F(BACKPROP_TOOLS_RL_CUDA, VIEW_COPY_PROBLEM) {

    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    bpt::randn(device_cpu, batch_cpu.observations_actions_next_observations, rng_cpu);
    bpt::set_all(device_cpu, batch_cpu_2.observations_actions_next_observations, 0);
    bpt::copy(device_gpu, device_cpu, batch_gpu.next_observations, batch_cpu.next_observations);
    bpt::copy(device_cpu, device_gpu, batch_cpu_2.next_observations, batch_gpu.next_observations);

    auto abs_diff_next_observations = bpt::abs_diff(device_cpu, batch_cpu_2.next_observations, batch_cpu.next_observations);
    ASSERT_LT(abs_diff_next_observations, EPSILON);
    ASSERT_LT(bpt::sum(device_cpu, batch_cpu_2.observations), EPSILON);
    ASSERT_LT(bpt::sum(device_cpu, batch_cpu_2.actions), EPSILON);
}

TEST_F(BACKPROP_TOOLS_RL_CUDA, GATHER_BATCH) {

    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());
    for(DEVICE_CPU::index_t rb_i = 0; rb_i < OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS; rb_i++) {
        bpt::copy(device_cpu, device_gpu, off_policy_runner_cpu_2.replay_buffers[rb_i], off_policy_runner_gpu_cpu.replay_buffers[rb_i]);
        auto abs_diff = bpt::abs_diff(device_cpu, off_policy_runner_cpu.replay_buffers[rb_i], off_policy_runner_cpu_2.replay_buffers[rb_i]);
        ASSERT_FLOAT_EQ(abs_diff, 0);
    }

    bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
    bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

    auto batch_observation_view = bpt::view<DEVICE_CPU, typename decltype(off_policy_runner_cpu.replay_buffers[0].observations)::SPEC, BATCH_SIZE, BATCH_TYPE::OBSERVATION_DIM>(device_cpu, off_policy_runner_cpu.replay_buffers[0].observations, 0, 0);
    bpt::print(device_cpu, batch_observation_view);
    std::cout << "BATCH" << std::endl;
    bpt::print(device_cpu, batch_cpu.observations);
    std::cout << "BATCH GPU" << std::endl;
    bpt::copy(device_cpu, device_gpu, batch_cpu_2, batch_gpu);
    bpt::print(device_cpu, batch_cpu_2.observations);

    auto abs_diff_batch = bpt::abs_diff(device_cpu, batch_cpu.observations, batch_cpu_2.observations);
    abs_diff_batch += bpt::abs_diff(device_cpu, batch_cpu.actions, batch_cpu_2.actions);
    abs_diff_batch += bpt::abs_diff(device_cpu, batch_cpu.next_observations, batch_cpu_2.next_observations);
    abs_diff_batch += bpt::abs_diff(device_cpu, batch_cpu.rewards, batch_cpu_2.rewards);
    abs_diff_batch += bpt::abs_diff(device_cpu, batch_cpu.terminated, batch_cpu_2.terminated);
    abs_diff_batch += bpt::abs_diff(device_cpu, batch_cpu.truncated, batch_cpu_2.truncated);
    ASSERT_FLOAT_EQ(abs_diff_batch, 0);
}
TEST_F(BACKPROP_TOOLS_RL_CUDA, TRAIN_CRITIC_STEP_BY_STEP) {
    constexpr DEVICE_CPU::index_t N_STEPS = 5;

    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    auto sample_batch = [&](bool deterministic){
        rng_gpu = bpt::random::next(DEVICE_GPU::SPEC::RANDOM(), rng_gpu);
        if(deterministic){
            bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
            bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

            // action noise from cpu
            bpt::target_action_noise(device_cpu, actor_critic_cpu, critic_training_buffers_cpu.target_next_action_noise, rng_cpu);
            bpt::copy(device_gpu, device_cpu, critic_training_buffers_gpu.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto abs_diff_target_next_action_noise = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            std::cout << "abs_diff_target_next_action_noise: " << abs_diff_target_next_action_noise << std::endl;
            ASSERT_FLOAT_EQ(abs_diff_target_next_action_noise, 0);
        }
        else{
            bpt::gather_batch(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
            bpt::copy(device_cpu, device_gpu, batch_cpu, batch_gpu);

            // action noise from gpu
            bpt::target_action_noise(device_gpu, actor_critic_gpu, critic_training_buffers_gpu.target_next_action_noise, rng_gpu);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto action_noise_std = bpt::std(device_cpu, critic_training_buffers_cpu.target_next_action_noise);
            auto action_noise_std_diff = std::abs(action_noise_std - ACTOR_CRITIC_SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD);
            std::cout << "action_noise_std_diff: " << action_noise_std_diff << std::endl;
            ASSERT_LT(action_noise_std_diff, 0.05);
        }

//    bpt::target_action_noise(device_gpu, actor_critic_gpu, critic_training_buffers_gpu.target_next_action_noise, rng_gpu);

        static_assert(BATCH_SPEC::BATCH_SIZE == ACTOR_BUFFERS::BATCH_SIZE);

        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
        auto abs_diff_actor_critic = bpt::abs_diff(device_cpu, actor_critic_cpu_2.actor_target, actor_critic_cpu.actor_target);
        std::cout << "abs_diff_actor_critic: " << abs_diff_actor_critic << std::endl;
        ASSERT_FLOAT_EQ(abs_diff_actor_critic, 0);

        bpt::evaluate(device_cpu, actor_critic_cpu.actor.input_layer, batch_cpu.observations, actor_buffers_cpu.tick);
        bpt::evaluate(device_gpu, actor_critic_gpu.actor.input_layer, batch_gpu.observations, actor_buffers_gpu.tick);
        bpt::copy(device_cpu, device_gpu, actor_buffers_cpu_2, actor_buffers_gpu);
        auto abs_diff_tick = bpt::abs_diff(device_cpu, actor_buffers_cpu_2.tick, actor_buffers_cpu.tick);
        std::cout << "abs_diff_tick: " << abs_diff_tick << std::endl;
        ASSERT_LT(abs_diff_tick, EPSILON);

        bpt::evaluate(device_cpu, actor_critic_cpu.actor_target, batch_cpu.next_observations, critic_training_buffers_cpu.next_actions, actor_buffers_cpu);
        bpt::evaluate(device_gpu, actor_critic_gpu.actor_target, batch_gpu.next_observations, critic_training_buffers_gpu.next_actions, actor_buffers_gpu);

        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        auto abs_diff_next_actions = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_actions, critic_training_buffers_cpu.next_actions);
        std::cout << "abs_diff_next_actions: " << abs_diff_next_actions << std::endl;
        ASSERT_LT(abs_diff_next_actions, EPSILON);

        bpt::noisy_next_actions(device_cpu, critic_training_buffers_cpu);
        bpt::noisy_next_actions(device_gpu, critic_training_buffers_gpu);
        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        bpt::check_status(device_gpu);
        auto abs_diff_noisy_next_actions = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_actions, critic_training_buffers_cpu.next_actions);
        std::cout << "abs_diff_noisy_next_actions: " << abs_diff_noisy_next_actions << std::endl;
        ASSERT_LT(abs_diff_noisy_next_actions, EPSILON);

        bpt::copy(device_cpu, device_cpu, critic_training_buffers_cpu.next_observations, batch_cpu.next_observations);
        bpt::copy(device_gpu, device_gpu, critic_training_buffers_gpu.next_observations, batch_gpu.next_observations);
        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        auto abs_diff_next_state_action_value_input = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_state_action_value_input, critic_training_buffers_cpu.next_state_action_value_input);
        std::cout << "abs_diff_next_state_action_value_input: " << abs_diff_next_state_action_value_input << std::endl;
        ASSERT_LT(abs_diff_next_state_action_value_input, EPSILON);

        bpt::evaluate(device_cpu, actor_critic_cpu.critic_target_1, critic_training_buffers_cpu.next_state_action_value_input, critic_training_buffers_cpu.next_state_action_value_critic_1, critic_buffers_cpu);
        bpt::evaluate(device_cpu, actor_critic_cpu.critic_target_2, critic_training_buffers_cpu.next_state_action_value_input, critic_training_buffers_cpu.next_state_action_value_critic_2, critic_buffers_cpu);

        bpt::evaluate(device_gpu, actor_critic_gpu.critic_target_1, critic_training_buffers_gpu.next_state_action_value_input, critic_training_buffers_gpu.next_state_action_value_critic_1, critic_buffers_gpu);
        bpt::evaluate(device_gpu, actor_critic_gpu.critic_target_2, critic_training_buffers_gpu.next_state_action_value_input, critic_training_buffers_gpu.next_state_action_value_critic_2, critic_buffers_gpu);

        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        auto abs_diff_next_state_action_value_critic_1 = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_state_action_value_critic_1, critic_training_buffers_cpu.next_state_action_value_critic_1);
        auto abs_diff_next_state_action_value_critic_2 = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.next_state_action_value_critic_2, critic_training_buffers_cpu.next_state_action_value_critic_2);
        std::cout << "abs_diff_next_state_action_value_critic_1: " << abs_diff_next_state_action_value_critic_1 << std::endl;
        std::cout << "abs_diff_next_state_action_value_critic_2: " << abs_diff_next_state_action_value_critic_2 << std::endl;
        ASSERT_LT(abs_diff_next_state_action_value_critic_1, EPSILON);
        ASSERT_LT(abs_diff_next_state_action_value_critic_2, EPSILON);

        bpt::target_actions(device_cpu, batch_cpu, critic_training_buffers_cpu);
        bpt::target_actions(device_gpu, batch_gpu, critic_training_buffers_gpu);
        bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2, critic_training_buffers_gpu);
        auto abs_diff_target_action_value = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.target_action_value, critic_training_buffers_cpu.target_action_value);
        std::cout << "abs_diff_target_action_value: " << abs_diff_target_action_value << std::endl;
        ASSERT_LT(abs_diff_target_action_value, EPSILON);
    };

    sample_batch(true);

    for(typename DEVICE_CPU::index_t i = 0; i < N_STEPS; i++){
        typename DEVICE_CPU::index_t critic_i = i % 2;
        sample_batch(false);
        auto& critic_cpu = critic_i == 0 ? actor_critic_cpu.critic_1 : actor_critic_cpu.critic_2;
        auto& critic_gpu = critic_i == 0 ? actor_critic_gpu.critic_1 : actor_critic_gpu.critic_2;
        auto& critic_cpu_2 = critic_i == 0 ? actor_critic_cpu_2.critic_1 : actor_critic_cpu_2.critic_2;

        bpt::zero_gradient(device_cpu, critic_cpu);
        bpt::zero_gradient(device_gpu, critic_gpu);

//        forward_backward_mse(device_cpu, critic_cpu, batch_cpu.observations_and_actions, critic_training_buffers_cpu.target_action_value, critic_buffers_cpu);
        {
            bpt::forward(device_cpu, critic_cpu, batch_cpu.observations_and_actions);
            bpt::nn::loss_functions::mse::gradient(device_cpu, output(critic_cpu), critic_training_buffers_cpu.target_action_value, d_critic_output_cpu);
            bpt::backward(device_cpu, critic_cpu, batch_cpu.observations_and_actions, d_critic_output_cpu, critic_buffers_cpu);
        }
//        forward_backward_mse(device_gpu, critic_gpu, batch_gpu.observations_and_actions, critic_training_buffers_gpu.target_action_value, critic_buffers_gpu);
        {
            bpt::forward(device_gpu, critic_gpu, batch_gpu.observations_and_actions);
            bpt::nn::loss_functions::mse::gradient(device_gpu, output(critic_gpu), critic_training_buffers_gpu.target_action_value, d_critic_output_gpu);
            bpt::backward(device_gpu, critic_gpu, batch_gpu.observations_and_actions, d_critic_output_gpu, critic_buffers_gpu);
        }
        bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);

        auto abs_diff_critic = bpt::abs_diff(device_cpu, critic_cpu, critic_cpu_2);
        std::cout << "abs_diff_critic: " << abs_diff_critic << std::endl;
        ASSERT_LT(abs_diff_critic, EPSILON);

        bpt::step(device_cpu, actor_critic_cpu.critic_optimizers[0], critic_cpu);
        bpt::step(device_gpu, actor_critic_gpu.critic_optimizers[0], critic_gpu);
        bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
        auto abs_diff_critic_after_update = bpt::abs_diff(device_cpu, critic_cpu, critic_cpu_2);
        std::cout << "abs_diff_critic_after_update: " << abs_diff_critic_after_update << std::endl;
        ASSERT_LT(abs_diff_critic_after_update, EPSILON);

        if(i % 5 == 0){
            bpt::update_critic_targets(device_cpu, actor_critic_cpu);
            bpt::update_critic_targets(device_gpu, actor_critic_gpu);
            bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
            auto abs_diff_critic_target_1 = bpt::abs_diff(device_cpu, actor_critic_cpu.critic_target_1, actor_critic_cpu_2.critic_target_1);
            auto abs_diff_critic_target_2 = bpt::abs_diff(device_cpu, actor_critic_cpu.critic_target_2, actor_critic_cpu_2.critic_target_2);
            std::cout << "abs_diff_critic_target_1: " << abs_diff_critic_target_1 << std::endl;
            std::cout << "abs_diff_critic_target_2: " << abs_diff_critic_target_2 << std::endl;
            ASSERT_LT(abs_diff_critic_target_1, EPSILON);
            ASSERT_LT(abs_diff_critic_target_2, EPSILON);
        }
    }


}

TEST_F(BACKPROP_TOOLS_RL_CUDA, TRAIN_CRITIC_CORRECTNESS) {
    constexpr DEVICE_CPU::index_t N_STEPS = 50;

    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    auto sample_batch = [&](bool deterministic){
        rng_gpu = bpt::random::next(DEVICE_GPU::SPEC::RANDOM(), rng_gpu);
        if(deterministic){
            bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
            bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

            // action noise from cpu
            bpt::target_action_noise(device_cpu, actor_critic_cpu, critic_training_buffers_cpu.target_next_action_noise, rng_cpu);
            bpt::copy(device_gpu, device_cpu, critic_training_buffers_gpu.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto abs_diff_target_next_action_noise = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            std::cout << "abs_diff_target_next_action_noise: " << abs_diff_target_next_action_noise << std::endl;
            ASSERT_FLOAT_EQ(abs_diff_target_next_action_noise, 0);
        }
        else{
            bpt::gather_batch(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
            bpt::copy(device_cpu, device_gpu, batch_cpu, batch_gpu);

            // action noise from gpu
            bpt::target_action_noise(device_gpu, actor_critic_gpu, critic_training_buffers_gpu.target_next_action_noise, rng_gpu);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto action_noise_std = bpt::std(device_cpu, critic_training_buffers_cpu.target_next_action_noise);
            auto action_noise_std_diff = std::abs(action_noise_std - ACTOR_CRITIC_SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD);
            std::cout << "action_noise_std_diff: " << action_noise_std_diff << std::endl;
            ASSERT_LT(action_noise_std_diff, 0.05);
        }
    };

    sample_batch(true);

    for(typename DEVICE_CPU::index_t i = 0; i < N_STEPS; i++){
        typename DEVICE_CPU::index_t critic_i = i % 2;
        sample_batch(false);
        auto& critic_cpu = critic_i == 0 ? actor_critic_cpu.critic_1 : actor_critic_cpu.critic_2;
        auto& critic_gpu = critic_i == 0 ? actor_critic_gpu.critic_1 : actor_critic_gpu.critic_2;
        auto& critic_cpu_2 = critic_i == 0 ? actor_critic_cpu_2.critic_1 : actor_critic_cpu_2.critic_2;

        bpt::train_critic(device_cpu, actor_critic_cpu, critic_cpu, batch_cpu, actor_critic_cpu.critic_optimizers[0], actor_buffers_cpu, critic_buffers_cpu, critic_training_buffers_cpu);
        bpt::train_critic(device_gpu, actor_critic_gpu, critic_gpu, batch_gpu, actor_critic_gpu.critic_optimizers[0], actor_buffers_gpu, critic_buffers_gpu, critic_training_buffers_gpu);

        bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
        auto abs_diff_critic_after_update = bpt::abs_diff(device_cpu, critic_cpu, critic_cpu_2);
        std::cout << "abs_diff_critic_after_update: " << abs_diff_critic_after_update << std::endl;
        ASSERT_LT(abs_diff_critic_after_update, EPSILON);

        if(i % 5 == 0){
            bpt::update_critic_targets(device_cpu, actor_critic_cpu);
            bpt::update_critic_targets(device_gpu, actor_critic_gpu);
            bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
            auto abs_diff_critic_target_1 = bpt::abs_diff(device_cpu, actor_critic_cpu.critic_target_1, actor_critic_cpu_2.critic_target_1);
            auto abs_diff_critic_target_2 = bpt::abs_diff(device_cpu, actor_critic_cpu.critic_target_2, actor_critic_cpu_2.critic_target_2);
            std::cout << "abs_diff_critic_target_1: " << abs_diff_critic_target_1 << std::endl;
            std::cout << "abs_diff_critic_target_2: " << abs_diff_critic_target_2 << std::endl;
            ASSERT_LT(abs_diff_critic_target_1, EPSILON);
            ASSERT_LT(abs_diff_critic_target_2, EPSILON);
        }
    }
}

TEST_F(BACKPROP_TOOLS_RL_CUDA, TRAIN_CRITIC_PERFORMANCE) {
    using DEVICE_MKL = bpt::devices::DefaultCPU_MKL;
    DEVICE_MKL device_mkl;
    constexpr DEVICE_CPU::index_t N_STEPS = 10000;

    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    auto sample_batch = [&](bool deterministic){
        rng_gpu = bpt::random::next(DEVICE_GPU::SPEC::RANDOM(), rng_gpu);
        if(deterministic){
            bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
            bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);

            // action noise from cpu
            bpt::target_action_noise(device_cpu, actor_critic_cpu, critic_training_buffers_cpu.target_next_action_noise, rng_cpu);
            bpt::copy(device_gpu, device_cpu, critic_training_buffers_gpu.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto abs_diff_target_next_action_noise = bpt::abs_diff(device_cpu, critic_training_buffers_cpu_2.target_next_action_noise, critic_training_buffers_cpu.target_next_action_noise);
            std::cout << "abs_diff_target_next_action_noise: " << abs_diff_target_next_action_noise << std::endl;
            ASSERT_FLOAT_EQ(abs_diff_target_next_action_noise, 0);
        }
        else{
            bpt::gather_batch(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
            bpt::copy(device_cpu, device_gpu, batch_cpu, batch_gpu);

            // action noise from gpu
            bpt::target_action_noise(device_gpu, actor_critic_gpu, critic_training_buffers_gpu.target_next_action_noise, rng_gpu);
            bpt::copy(device_cpu, device_gpu, critic_training_buffers_cpu.target_next_action_noise, critic_training_buffers_gpu.target_next_action_noise);
            auto action_noise_std = bpt::std(device_cpu, critic_training_buffers_cpu.target_next_action_noise);
            auto action_noise_std_diff = std::abs(action_noise_std - ACTOR_CRITIC_SPEC::PARAMETERS::TARGET_NEXT_ACTION_NOISE_STD);
            std::cout << "action_noise_std_diff: " << action_noise_std_diff << std::endl;
            ASSERT_LT(action_noise_std_diff, 0.05);
        }
    };

    sample_batch(true);

    {
        auto& critic_cpu = actor_critic_cpu.critic_1;
        auto start = std::chrono::high_resolution_clock::now();
        for(typename DEVICE_CPU::index_t i = 0; i < N_STEPS; i++){
            bpt::train_critic(device_mkl, actor_critic_cpu, critic_cpu, batch_cpu, actor_critic_cpu.critic_optimizers[0], actor_buffers_cpu, critic_buffers_cpu, critic_training_buffers_cpu);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CPU train_critic: " << duration.count()/N_STEPS << " microseconds" << std::endl;
    }
    {
        auto& critic_gpu = actor_critic_gpu.critic_1;
        cudaDeviceSynchronize();
        auto start = std::chrono::high_resolution_clock::now();
        for(typename DEVICE_CPU::index_t i = 0; i < N_STEPS; i++){
            bpt::train_critic(device_gpu, actor_critic_gpu, critic_gpu, batch_gpu, actor_critic_gpu.critic_optimizers[0], actor_buffers_gpu, critic_buffers_gpu, critic_training_buffers_gpu);
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "GPU train_critic: " << duration.count()/N_STEPS << " microseconds" << std::endl;
    }
}

TEST_F(BACKPROP_TOOLS_RL_CUDA, TRAIN_ACTOR_CORRECTNESS) {
    constexpr DEVICE_CPU::index_t N_STEPS = 50;
    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    auto sample_batch = [&](bool deterministic){
        rng_gpu = bpt::random::next(DEVICE_GPU::SPEC::RANDOM(), rng_gpu);
        if(deterministic){
            bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
            bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
        }
        else{
            bpt::gather_batch(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
            bpt::copy(device_cpu, device_gpu, batch_cpu, batch_gpu);
        }
    };
    for(typename DEVICE_CPU::index_t step_i = 0; step_i < N_STEPS; step_i++){
        sample_batch(false);

        bpt::train_actor(device_cpu, actor_critic_cpu, batch_cpu, actor_critic_cpu.actor_optimizer, actor_buffers_cpu, critic_buffers_cpu, actor_training_buffers_cpu);
        bpt::train_actor(device_gpu, actor_critic_gpu, batch_gpu, actor_critic_gpu.actor_optimizer, actor_buffers_gpu, critic_buffers_gpu, actor_training_buffers_gpu);

        bpt::copy(device_cpu, device_gpu, actor_critic_cpu_2, actor_critic_gpu);
        auto abs_diff_actor_after_update = bpt::abs_diff(device_cpu, actor_critic_cpu.actor, actor_critic_cpu_2.actor);
        std::cout << "abs_diff_actor_after_update: " << abs_diff_actor_after_update << std::endl;
        ASSERT_LT(abs_diff_actor_after_update, EPSILON);
    }

}

TEST_F(BACKPROP_TOOLS_RL_CUDA, TRAIN_ACTOR_PERFORMANCE) {
    constexpr DEVICE_CPU::index_t N_STEPS = 10000;
    auto rng_cpu = bpt::random::default_engine(DEVICE_CPU::SPEC::RANDOM());
    auto rng_gpu = bpt::random::default_engine(DEVICE_GPU::SPEC::RANDOM());

    auto sample_batch = [&](bool deterministic){
        rng_gpu = bpt::random::next(DEVICE_GPU::SPEC::RANDOM(), rng_gpu);
        if(deterministic){
            bpt::gather_batch<DEVICE_CPU, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_cpu), true>(device_cpu, off_policy_runner_cpu, batch_cpu, rng_cpu);
            bpt::gather_batch<typename DEVICE_GPU::SPEC, OFF_POLICY_RUNNER_SPEC, BATCH_SPEC, decltype(rng_gpu), true>(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
        }
        else{
            bpt::gather_batch(device_gpu, off_policy_runner_gpu_struct, batch_gpu_struct, rng_gpu);
            bpt::copy(device_cpu, device_gpu, batch_cpu, batch_gpu);
        }
    };
    sample_batch(false);

    auto start = std::chrono::high_resolution_clock::now();
    for(typename DEVICE_CPU::index_t step_i = 0; step_i < N_STEPS; step_i++){
        bpt::train_actor(device_cpu, actor_critic_cpu, batch_cpu, actor_critic_cpu.actor_optimizer, actor_buffers_cpu, critic_buffers_cpu, actor_training_buffers_cpu);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "CPU time: " << duration.count()/N_STEPS << " microseconds" << std::endl;

    cudaDeviceSynchronize();
    start = std::chrono::high_resolution_clock::now();
    for(typename DEVICE_CPU::index_t step_i = 0; step_i < N_STEPS; step_i++){
        bpt::train_actor(device_gpu, actor_critic_gpu, batch_gpu, actor_critic_gpu.actor_optimizer, actor_buffers_gpu, critic_buffers_gpu, actor_training_buffers_gpu);
    }
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "GPU time: " << duration.count()/N_STEPS << " microseconds" << std::endl;
}
