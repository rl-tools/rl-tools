#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/batch/operations_cuda.h>

#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>

#include <rl_tools/rl/algorithms/ppo/operations_cuda.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_cuda.h>

#include <rl_tools/random/operations_generic_array.h>

#include <gtest/gtest.h>
#include <iostream>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

// --- Device types ---
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE_CPU::index_t;

// --- Environment ---
using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

// --- PPO hyperparameters ---
constexpr TI N_ENVIRONMENTS = 4;
constexpr TI STEPS_PER_ENV = 64;
constexpr TI BATCH_SIZE = N_ENVIRONMENTS * STEPS_PER_ENV;
using BATCH_SPEC = rlt::rl::environments::batch::Specification<ENVIRONMENT, N_ENVIRONMENTS>;
using BATCH = rlt::rl::environments::batch::Independent<BATCH_SPEC>;

struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
    static constexpr TI N_EPOCHS = 2;
    static constexpr bool LEARN_ACTION_STD = true;
    static constexpr T INITIAL_ACTION_STD = 0.5;
    static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
    static constexpr bool NORMALIZE_ADVANTAGE = true;
    static constexpr T GAMMA = 0.99;
    static constexpr bool SHUFFLE_EPOCH = false;
    static constexpr bool ADAPTIVE_LEARNING_RATE = false;
};

// --- Actor (compiled for BATCH_SIZE training, eval buffers for N_ENVIRONMENTS) ---
using ACTOR_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
struct ActorBuilder{
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, ENVIRONMENT::Observation::DIM>;
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
    using MODULE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
    using CHAIN = rlt::nn_models::sequential::Module<MODULE>;
    using MODEL = rlt::nn_models::sequential::Build<ACTOR_CAPABILITY, CHAIN, INPUT_SHAPE>;
};
using ACTOR_TYPE = ActorBuilder::MODEL;

// --- Critic ---
struct CriticBuilder{
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, BATCH_SIZE, ENVIRONMENT::ObservationPrivileged::DIM>;
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
    using MODULE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
    using CHAIN = rlt::nn_models::sequential::Module<MODULE>;
    using MODEL = rlt::nn_models::sequential::Build<ACTOR_CAPABILITY, CHAIN, INPUT_SHAPE>;
};
using CRITIC_TYPE = CriticBuilder::MODEL;

// --- PPO type ---
using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<TYPE_POLICY, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
using PPO_TYPE = rlt::rl::algorithms::PPO<PPO_SPEC>;
using PPO_BUFFERS_SPEC = rlt::rl::algorithms::ppo::BufferSpecification<PPO_SPEC>;
using PPO_BUFFERS = rlt::rl::algorithms::ppo::Buffers<PPO_BUFFERS_SPEC>;

// --- On-policy runner ---
using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, BATCH, typename ACTOR_TYPE::template State<>>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

// --- Buffers ---
using ACTOR_EVAL_TYPE = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>;
using ACTOR_EVAL_BUFFERS = typename ACTOR_EVAL_TYPE::template Buffer<>;
using ACTOR_TRAIN_BUFFERS = typename ACTOR_TYPE::template Buffer<>;
using CRITIC_TRAIN_BUFFERS = typename CRITIC_TYPE::template Buffer<>;
using CRITIC_GAE_TYPE = typename CRITIC_TYPE::template CHANGE_BATCH_SIZE<TI, DATASET_SPEC::STEPS_TOTAL_ALL>;
using CRITIC_GAE_BUFFERS = typename CRITIC_GAE_TYPE::template Buffer<>;

// --- Optimizers ---
using ACTOR_OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>;
using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ACTOR_OPTIMIZER_PARAMETERS>;
using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<TYPE_POLICY, TI, ACTOR_OPTIMIZER_PARAMETERS>;
using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;

// --- Array RNG ---
using ARRAY_RNG_SPEC = rlt::devices::generic::random::ArraySpecification<TI, 1024>;
using ARRAY_RNG = rlt::devices::generic::random::ArrayENGINE<ARRAY_RNG_SPEC>;

// Run critic evaluate to fill all_values (mirrors PPO loop core step)
template <typename DEVICE, typename PPO_TYPE_T, typename DATASET_T, typename CRITIC_GAE_BUFFER_T, typename RNG_T>
void evaluate_critic_for_gae(DEVICE& device, PPO_TYPE_T& ppo, DATASET_T& dataset, CRITIC_GAE_BUFFER_T& critic_buffers_gae, RNG_T& rng){
    using TI_INNER = typename DATASET_T::TI;
    constexpr TI_INNER STEPS_TOTAL_ALL = DATASET_T::DATASET_SPEC::STEPS_TOTAL_ALL;
    using OBS_PRIV_SHAPE = typename DATASET_T::OBS_PRIV_SHAPE;
    using CRITIC_GAE_INPUT_SHAPE = rlt::tensor::Prepend<rlt::tensor::Prepend<OBS_PRIV_SHAPE, STEPS_TOTAL_ALL>, 1>;
    auto all_observations_privileged_reshaped = rlt::reshape_row_major(device, dataset.all_observations_privileged, CRITIC_GAE_INPUT_SHAPE{});
    auto all_values_tensor = rlt::to_tensor(device, dataset.all_values);
    auto all_values_tensor_reshaped = rlt::reshape_row_major(device, all_values_tensor, rlt::tensor::Shape<TI_INNER, 1, STEPS_TOTAL_ALL, 1>{});
    rlt::Mode<rlt::mode::Evaluation<>> mode;
    rlt::evaluate(device, ppo.critic, all_observations_privileged_reshaped, all_values_tensor_reshaped, critic_buffers_gae, rng, mode);
}

// Debug kernel: sample a random number with PortableState and write it to output
template <typename DEVICE>
__global__
void debug_rng_kernel(DEVICE device, rlt::devices::generic::random::PortableState* states, float* output, unsigned int n){
    unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        auto& rng_state = states[i];
        output[i] = rlt::random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (float)-3.14159f, (float)3.14159f, rng_state);
    }
}

TEST(RL_TOOLS_RL_ALGORITHMS_PPO_CUDA, RNG_SANITY_CHECK){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);

    constexpr unsigned int N = 4;
    ARRAY_RNG rng_cpu;
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 137);

    // CPU: sample N values using per-element RNG
    float cpu_values[N];
    for(unsigned int i = 0; i < N; i++){
        auto& rng_state = rlt::get(rng_cpu.states, 0, i);
        cpu_values[i] = rlt::random::uniform_real_distribution(typename DEVICE_CPU::SPEC::RANDOM(), -3.14159f, 3.14159f, rng_state);
    }

    // Reset RNG
    rlt::init(device_cpu, rng_cpu, 137);

    // GPU: allocate states and output on device
    rlt::devices::generic::random::PortableState* d_states;
    float* d_output;
    cudaMalloc(&d_states, N * sizeof(rlt::devices::generic::random::PortableState));
    cudaMalloc(&d_output, N * sizeof(float));
    // Copy CPU states to GPU
    rlt::devices::generic::random::PortableState h_states[N];
    for(unsigned int i = 0; i < N; i++){
        h_states[i] = rlt::get(rng_cpu.states, 0, i);
    }
    cudaMemcpy(d_states, h_states, N * sizeof(rlt::devices::generic::random::PortableState), cudaMemcpyHostToDevice);

    rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
    debug_rng_kernel<<<1, N, 0, device_gpu.stream>>>(tag_device, d_states, d_output, N);
    cudaDeviceSynchronize();

    float gpu_values[N];
    cudaMemcpy(gpu_values, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "RNG sanity check:" << std::endl;
    for(unsigned int i = 0; i < N; i++){
        std::cout << "  [" << i << "] CPU=" << cpu_values[i] << " GPU=" << gpu_values[i] << " diff=" << std::abs(cpu_values[i] - gpu_values[i]) << std::endl;
    }
    for(unsigned int i = 0; i < N; i++){
        EXPECT_NEAR(cpu_values[i], gpu_values[i], 1e-5) << "RNG mismatch at index " << i;
    }

    cudaFree(d_states);
    cudaFree(d_output);
    rlt::free(device_cpu, rng_cpu);
}

TEST(RL_TOOLS_RL_ALGORITHMS_PPO_CUDA, E2E_CPU_GPU_COMPARISON){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);

    constexpr TI NUM_STEPS = 200;
    constexpr TI SYNC_INTERVAL = 1;

    // --- Allocate ---
    PPO_TYPE ppo_cpu, ppo_gpu;
    ACTOR_OPTIMIZER actor_optimizer_cpu, actor_optimizer_gpu;
    CRITIC_OPTIMIZER critic_optimizer_cpu, critic_optimizer_gpu;
    PPO_BUFFERS ppo_buffers_cpu, ppo_buffers_gpu;
    ACTOR_EVAL_BUFFERS actor_eval_buffers_cpu, actor_eval_buffers_gpu;
    ACTOR_TRAIN_BUFFERS actor_train_buffers_cpu, actor_train_buffers_gpu;
    CRITIC_TRAIN_BUFFERS critic_train_buffers_cpu, critic_train_buffers_gpu;
    CRITIC_GAE_BUFFERS critic_gae_buffers_cpu, critic_gae_buffers_gpu;
    DATASET dataset_cpu, dataset_gpu;
    rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC> runner_cpu, runner_gpu;
    rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC> runner_buffer_cpu, runner_buffer_gpu;
    BATCH environment_cpu, environment_gpu;
    ARRAY_RNG rng_cpu, rng_gpu;
    DATASET dataset_gpu_copy;
    PPO_TYPE ppo_gpu_copy;

    rlt::malloc(device_cpu, ppo_cpu);
    rlt::malloc(device_cpu, actor_optimizer_cpu);
    rlt::malloc(device_cpu, critic_optimizer_cpu);
    rlt::malloc(device_cpu, ppo_buffers_cpu);
    rlt::malloc(device_cpu, actor_eval_buffers_cpu);
    rlt::malloc(device_cpu, actor_train_buffers_cpu);
    rlt::malloc(device_cpu, critic_train_buffers_cpu);
    rlt::malloc(device_cpu, critic_gae_buffers_cpu);
    rlt::malloc(device_cpu, dataset_cpu);
    rlt::malloc(device_cpu, runner_cpu);
    rlt::malloc(device_cpu, runner_buffer_cpu);
    rlt::malloc(device_cpu, environment_cpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::malloc(device_cpu, dataset_gpu_copy);
    rlt::malloc(device_cpu, ppo_gpu_copy);

    rlt::malloc(device_gpu, ppo_gpu);
    rlt::malloc(device_gpu, actor_optimizer_gpu);
    rlt::malloc(device_gpu, critic_optimizer_gpu);
    rlt::malloc(device_gpu, ppo_buffers_gpu);
    rlt::malloc(device_gpu, actor_eval_buffers_gpu);
    rlt::malloc(device_gpu, actor_train_buffers_gpu);
    rlt::malloc(device_gpu, critic_train_buffers_gpu);
    rlt::malloc(device_gpu, critic_gae_buffers_gpu);
    rlt::malloc(device_gpu, dataset_gpu);
    rlt::malloc(device_gpu, runner_gpu);
    rlt::malloc(device_gpu, runner_buffer_gpu);
    rlt::malloc(device_gpu, environment_gpu);
    rlt::malloc(device_gpu, rng_gpu);

    // --- Init PPO on CPU, copy to GPU ---
    typename DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_init;
    rlt::init(device_cpu, rng_init, 42);
    rlt::init(device_cpu, ppo_cpu, actor_optimizer_cpu, critic_optimizer_cpu, rng_init);
    rlt::copy(device_cpu, device_gpu, ppo_cpu, ppo_gpu);
    cudaDeviceSynchronize();

    // Verify actor weights match
    rlt::copy(device_gpu, device_cpu, ppo_gpu, ppo_gpu_copy);
    T weight_diff = rlt::abs_diff(device_cpu, ppo_cpu, ppo_gpu_copy);
    std::cout << "Initial weight diff (should be 0): " << weight_diff << std::endl;
    ASSERT_LT(weight_diff, 1e-6) << "PPO weights did not copy correctly";

    rlt::init(device_gpu, actor_optimizer_gpu);
    rlt::init(device_gpu, critic_optimizer_gpu);
    rlt::reset_optimizer_state(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
    rlt::reset_optimizer_state(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);

    // --- Init array RNG on CPU, copy to GPU ---
    rlt::init(device_cpu, rng_cpu, 137);
    rlt::copy(device_cpu, device_gpu, rng_cpu, rng_gpu);
    cudaDeviceSynchronize();

    // Verify RNG match
    ARRAY_RNG rng_gpu_check;
    rlt::malloc(device_cpu, rng_gpu_check);
    rlt::copy(device_gpu, device_cpu, rng_gpu, rng_gpu_check);
    auto rng_diff = rlt::abs_diff(device_cpu, rng_cpu, rng_gpu_check);
    std::cout << "Initial RNG diff (should be 0): " << rng_diff << std::endl;
    ASSERT_EQ(rng_diff, 0) << "RNG states did not copy correctly";
    rlt::free(device_cpu, rng_gpu_check);

    // --- Init runners identically ---
    rlt::init(device_cpu, environment_cpu);
    rlt::init(device_gpu, environment_gpu);
    rlt::init(device_cpu, runner_cpu, environment_cpu, rng_cpu);
    rlt::init(device_gpu, runner_gpu, environment_gpu, rng_gpu);
    cudaDeviceSynchronize();

    rlt::set_all(device_cpu, dataset_cpu.scalar_data, 0);
    rlt::set_all(device_gpu, dataset_gpu.scalar_data, 0);

    T max_collect_diff = 0;
    T max_gae_diff = 0;
    T max_train_diff = 0;

    // Element counts for per-element diff
    constexpr TI COLLECT_ELEMENTS =
        DATASET_SPEC::STEPS_TOTAL_ALL * ENVIRONMENT::Observation::DIM +
        DATASET_SPEC::STEPS_TOTAL_ALL * ENVIRONMENT::ObservationPrivileged::DIM +
        DATASET_SPEC::STEPS_TOTAL * ENVIRONMENT::ACTION_DIM * 2 +
        DATASET_SPEC::STEPS_TOTAL * 5 +
        DATASET_SPEC::STEPS_TOTAL_ALL * 3 +
        DATASET_SPEC::STEPS_TOTAL * 2;
    constexpr TI GAE_ELEMENTS = 2 * DATASET_SPEC::STEPS_TOTAL;

    // --- Training loop ---
    for(TI step = 0; step < NUM_STEPS; step++){
        // Synchronize CPU→GPU periodically to prevent drift
        if(step > 0 && step % SYNC_INTERVAL == 0){
            rlt::copy(device_cpu, device_gpu, ppo_cpu, ppo_gpu);
            rlt::copy(device_cpu, device_gpu, rng_cpu, rng_gpu);
            rlt::reset_optimizer_state(device_gpu, actor_optimizer_gpu, ppo_gpu.actor);
            rlt::reset_optimizer_state(device_cpu, actor_optimizer_cpu, ppo_cpu.actor);
            rlt::reset_optimizer_state(device_gpu, critic_optimizer_gpu, ppo_gpu.critic);
            rlt::reset_optimizer_state(device_cpu, critic_optimizer_cpu, ppo_cpu.critic);
            // Sync runner state via matrix copy (not element-wise set on GPU memory)
            rlt::copy(device_cpu, device_gpu, runner_cpu.episode_step, runner_gpu.episode_step);
            rlt::copy(device_cpu, device_gpu, runner_cpu.reset, runner_gpu.reset);
            rlt::copy(device_cpu, device_gpu, runner_cpu.episode_return, runner_gpu.episode_return);
            rlt::copy(device_cpu, device_gpu, runner_cpu.completed_episode_length, runner_gpu.completed_episode_length);
            rlt::copy(device_cpu, device_gpu, runner_cpu.completed_episode_return, runner_gpu.completed_episode_return);
            rlt::copy(device_cpu, device_gpu, runner_cpu.completed_episode_reason, runner_gpu.completed_episode_reason);
            rlt::copy(device_cpu, device_gpu, runner_cpu.states, runner_gpu.states);
            rlt::copy(device_cpu, device_gpu, environment_cpu.environments, environment_gpu.environments);
            rlt::copy(device_cpu, device_gpu, runner_cpu.env_parameters, runner_gpu.env_parameters);
            rlt::copy(device_cpu, device_gpu, runner_cpu.policy_state, runner_gpu.policy_state);
            runner_gpu.episode_step_limit = runner_cpu.episode_step_limit;
            runner_gpu.step = runner_cpu.step;
            cudaDeviceSynchronize();
        }

        // 1. Collect
        rlt::collect(device_cpu, dataset_cpu, runner_cpu, runner_buffer_cpu, environment_cpu, ppo_cpu.actor, actor_eval_buffers_cpu, rng_cpu);
        rlt::collect(device_gpu, dataset_gpu, runner_gpu, runner_buffer_gpu, environment_gpu, ppo_gpu.actor, actor_eval_buffers_gpu, rng_gpu);
        cudaDeviceSynchronize();

        rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations, dataset_gpu_copy.all_observations);
        rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations_privileged, dataset_gpu_copy.all_observations_privileged);
        rlt::copy(device_gpu, device_cpu, dataset_gpu.episode_end_reason, dataset_gpu_copy.episode_end_reason);
        rlt::copy(device_gpu, device_cpu, dataset_gpu.episode_length, dataset_gpu_copy.episode_length);
        rlt::copy(device_gpu, device_cpu, dataset_gpu.episode_return, dataset_gpu_copy.episode_return);
        rlt::copy(device_gpu, device_cpu, dataset_gpu.scalar_data, dataset_gpu_copy.scalar_data);
        T collect_diff = rlt::abs_diff(device_cpu, dataset_cpu, dataset_gpu_copy);
        if(collect_diff > max_collect_diff) max_collect_diff = collect_diff;

        if(step == 0){
            // Debug: print first few observations
            std::cout << "CPU obs[0]: ";
            auto cpu_obs = rlt::matrix_view(device_cpu, rlt::view(device_cpu, dataset_cpu.all_observations, (TI)0));
            for(TI j = 0; j < ENVIRONMENT::Observation::DIM; j++) std::cout << rlt::get(cpu_obs, 0, j) << " ";
            std::cout << std::endl;
            std::cout << "GPU obs[0]: ";
            auto gpu_obs = rlt::matrix_view(device_cpu, rlt::view(device_cpu, dataset_gpu_copy.all_observations, (TI)0));
            for(TI j = 0; j < ENVIRONMENT::Observation::DIM; j++) std::cout << rlt::get(gpu_obs, 0, j) << " ";
            std::cout << std::endl;
        }

        // 2. Evaluate critic (fill all_values)
        evaluate_critic_for_gae(device_cpu, ppo_cpu, dataset_cpu, critic_gae_buffers_cpu, rng_cpu);
        evaluate_critic_for_gae(device_gpu, ppo_gpu, dataset_gpu, critic_gae_buffers_gpu, rng_gpu);
        cudaDeviceSynchronize();

        // 3. GAE
        rlt::estimate_generalized_advantages(device_cpu, dataset_cpu, PPO_PARAMETERS{});
        rlt::estimate_generalized_advantages(device_gpu, dataset_gpu, PPO_PARAMETERS{});
        cudaDeviceSynchronize();

        rlt::copy(device_gpu, device_cpu, dataset_gpu.scalar_data, dataset_gpu_copy.scalar_data);
        T adv_diff = 0;
        for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL; i++){
            adv_diff += rlt::math::abs(device_cpu.math, rlt::get(dataset_cpu.advantages, i, 0) - rlt::get(dataset_gpu_copy.advantages, i, 0));
            adv_diff += rlt::math::abs(device_cpu.math, rlt::get(dataset_cpu.target_values, i, 0) - rlt::get(dataset_gpu_copy.target_values, i, 0));
        }
        if(adv_diff > max_gae_diff) max_gae_diff = adv_diff;

        // 4. Train
        rlt::train(device_cpu, ppo_cpu, dataset_cpu, actor_optimizer_cpu, critic_optimizer_cpu, ppo_buffers_cpu, actor_train_buffers_cpu, critic_train_buffers_cpu, rng_cpu);
        rlt::train(device_gpu, ppo_gpu, dataset_gpu, actor_optimizer_gpu, critic_optimizer_gpu, ppo_buffers_gpu, actor_train_buffers_gpu, critic_train_buffers_gpu, rng_gpu);
        cudaDeviceSynchronize();

        rlt::copy(device_gpu, device_cpu, ppo_gpu, ppo_gpu_copy);
        T train_diff = rlt::abs_diff(device_cpu, ppo_cpu, ppo_gpu_copy);
        if(train_diff > max_train_diff) max_train_diff = train_diff;

        if(step % 10 == 0 || step == NUM_STEPS - 1){
            std::cout << "Step " << step << "/" << NUM_STEPS
                      << " collect/el=" << collect_diff / COLLECT_ELEMENTS
                      << " gae/el=" << adv_diff / GAE_ELEMENTS
                      << " train/el=" << train_diff
                      << std::endl;
        }
    }

    std::cout << "\n=== Summary (per element) ===" << std::endl;
    std::cout << "Max collect diff/el: " << max_collect_diff / COLLECT_ELEMENTS << " (total: " << max_collect_diff << ", n=" << COLLECT_ELEMENTS << ")" << std::endl;
    std::cout << "Max GAE diff/el:     " << max_gae_diff / GAE_ELEMENTS << " (total: " << max_gae_diff << ", n=" << GAE_ELEMENTS << ")" << std::endl;
    std::cout << "Max train diff:      " << max_train_diff << std::endl;

    EXPECT_LT(max_collect_diff / COLLECT_ELEMENTS, 1e-12);
    EXPECT_LT(max_gae_diff / GAE_ELEMENTS, 1e-11);
    EXPECT_LT(max_train_diff, 1e-5);

    // --- Cleanup ---
    rlt::free(device_cpu, ppo_cpu);
    rlt::free(device_cpu, actor_optimizer_cpu);
    rlt::free(device_cpu, critic_optimizer_cpu);
    rlt::free(device_cpu, ppo_buffers_cpu);
    rlt::free(device_cpu, actor_eval_buffers_cpu);
    rlt::free(device_cpu, actor_train_buffers_cpu);
    rlt::free(device_cpu, critic_train_buffers_cpu);
    rlt::free(device_cpu, critic_gae_buffers_cpu);
    rlt::free(device_cpu, dataset_cpu);
    rlt::free(device_cpu, runner_cpu);
    rlt::free(device_cpu, runner_buffer_cpu);
    rlt::free(device_cpu, environment_cpu);
    rlt::free(device_cpu, rng_cpu);
    rlt::free(device_cpu, dataset_gpu_copy);
    rlt::free(device_cpu, ppo_gpu_copy);
    rlt::free(device_gpu, ppo_gpu);
    rlt::free(device_gpu, actor_optimizer_gpu);
    rlt::free(device_gpu, critic_optimizer_gpu);
    rlt::free(device_gpu, ppo_buffers_gpu);
    rlt::free(device_gpu, actor_eval_buffers_gpu);
    rlt::free(device_gpu, actor_train_buffers_gpu);
    rlt::free(device_gpu, critic_train_buffers_gpu);
    rlt::free(device_gpu, critic_gae_buffers_gpu);
    rlt::free(device_gpu, dataset_gpu);
    rlt::free(device_gpu, runner_gpu);
    rlt::free(device_gpu, runner_buffer_gpu);
    rlt::free(device_gpu, environment_gpu);
    rlt::free(device_gpu, rng_gpu);
}
