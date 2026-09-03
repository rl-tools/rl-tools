#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/batch/operations_cuda.h>

#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>

#include <gtest/gtest.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU = rlt::devices::DefaultCUDA;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE_CPU::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

constexpr TI N_ENVIRONMENTS = 4;
constexpr TI STEPS_PER_ENV = 64;
using BATCH_SPEC = rlt::rl::environments::batch::Specification<ENVIRONMENT, N_ENVIRONMENTS>;
using BATCH = rlt::rl::environments::batch::Independent<BATCH_SPEC>;

template <typename CAPABILITY>
struct ActorConfig{
    using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, N_ENVIRONMENTS, ENVIRONMENT::Observation::DIM>;
    using MLP_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
    using ACTOR_MODULE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<MLP_CONFIG>;
    using MODULE_CHAIN = rlt::nn_models::sequential::Module<ACTOR_MODULE>;
    using MODEL = rlt::nn_models::sequential::Build<CAPABILITY, MODULE_CHAIN, INPUT_SHAPE>;
};

using ACTOR_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam>;
using ACTOR_TYPE = typename ActorConfig<ACTOR_CAPABILITY>::MODEL;

using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, BATCH, ACTOR_TYPE::State<>>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;

using ON_POLICY_RUNNER = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
using ON_POLICY_RUNNER_BUFFER = rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

using ACTOR_BUFFERS = typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>::template Buffer<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<rlt::devices::random::CUDA::Specification<TI, 1024>>;


TEST(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_CUDA, COLLECT){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);

    // Initialize actor on CPU, then copy to GPU
    ACTOR_TYPE actor_cpu, actor_gpu;
    typename DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu;
    rlt::malloc(device_cpu, actor_cpu);
    rlt::malloc(device_cpu, rng_cpu);
    rlt::init(device_cpu, rng_cpu, 42);
    rlt::init_weights(device_cpu, actor_cpu, rng_cpu);

    rlt::malloc(device_gpu, actor_gpu);
    rlt::copy(device_cpu, device_gpu, actor_cpu, actor_gpu);

    // GPU runner and dataset
    ON_POLICY_RUNNER runner_gpu;
    ON_POLICY_RUNNER_BUFFER runner_buffer_gpu;
    BATCH environment_gpu;
    DATASET dataset_gpu;
    ACTOR_BUFFERS actor_buffers_gpu;
    RNG_GPU rng_gpu;

    rlt::malloc(device_gpu, runner_gpu);
    rlt::malloc(device_gpu, runner_buffer_gpu);
    rlt::malloc(device_gpu, environment_gpu);
    rlt::malloc(device_gpu, dataset_gpu);
    rlt::malloc(device_gpu, actor_buffers_gpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, 42);

    // Initialize runner on GPU
    rlt::init(device_gpu, environment_gpu);
    rlt::init(device_gpu, runner_gpu, environment_gpu, rng_gpu);

    rlt::set_all(device_gpu, dataset_gpu.scalar_data, 0);

    // Run collect on GPU
    rlt::collect(device_gpu, dataset_gpu, runner_gpu, runner_buffer_gpu, environment_gpu, actor_gpu, actor_buffers_gpu, rng_gpu);

    // Verify by copying individual tensors back to CPU
    rlt::Tensor<typename decltype(dataset_gpu.all_observations)::SPEC> obs_copy;
    rlt::Matrix<typename decltype(dataset_gpu.scalar_data)::SPEC> scalar_copy;
    rlt::malloc(device_cpu, obs_copy);
    rlt::malloc(device_cpu, scalar_copy);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations, obs_copy);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.scalar_data, scalar_copy);

    // Verify observations are non-zero (pendulum always produces non-trivial observations)
    T obs_l1 = 0;
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL_ALL; i++){
        for(TI j = 0; j < ENVIRONMENT::Observation::DIM; j++){
            obs_l1 += rlt::math::abs(device_cpu.math, rlt::get(device_cpu, obs_copy, i, j));
        }
    }
    std::cout << "Observations L1 norm: " << obs_l1 << std::endl;
    EXPECT_GT(obs_l1, 0) << "Observations should be non-zero";

    // Verify actions are non-zero (actor with random weights should produce non-trivial actions)
    // Actions are in the first ACTION_DIM columns of scalar_data
    T actions_l1 = 0;
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL; i++){
        // actions_mean are in columns [0, ACTION_DIM), actions in [ACTION_DIM, 2*ACTION_DIM)
        for(TI j = ENVIRONMENT::ACTION_DIM; j < 2 * ENVIRONMENT::ACTION_DIM; j++){
            actions_l1 += rlt::math::abs(device_cpu.math, rlt::get(scalar_copy, i, j));
        }
    }
    std::cout << "Actions L1 norm: " << actions_l1 << std::endl;
    EXPECT_GT(actions_l1, 0) << "Actions should be non-zero";

    // Verify rewards column is populated (rewards are at column offset 2*ACTION_DIM + 1)
    constexpr TI REWARD_COL = 2 * ENVIRONMENT::ACTION_DIM + 1;
    T rewards_l1 = 0;
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL; i++){
        rewards_l1 += rlt::math::abs(device_cpu.math, rlt::get(scalar_copy, i, REWARD_COL));
    }
    std::cout << "Rewards L1 norm: " << rewards_l1 << std::endl;
    EXPECT_GT(rewards_l1, 0) << "Rewards should be non-zero";

    // Run a second collect to verify state persistence
    rlt::collect(device_gpu, dataset_gpu, runner_gpu, runner_buffer_gpu, environment_gpu, actor_gpu, actor_buffers_gpu, rng_gpu);

    // Copy again and verify
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations, obs_copy);
    T obs_l1_2 = 0;
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL_ALL; i++){
        for(TI j = 0; j < ENVIRONMENT::Observation::DIM; j++){
            obs_l1_2 += rlt::math::abs(device_cpu.math, rlt::get(device_cpu, obs_copy, i, j));
        }
    }
    std::cout << "Observations L1 norm (2nd collect): " << obs_l1_2 << std::endl;
    EXPECT_GT(obs_l1_2, 0) << "Second collect should also produce non-zero observations";

    // Cleanup
    rlt::free(device_cpu, actor_cpu);
    rlt::free(device_gpu, actor_gpu);
    rlt::free(device_gpu, runner_gpu);
    rlt::free(device_gpu, runner_buffer_gpu);
    rlt::free(device_gpu, environment_gpu);
    rlt::free(device_gpu, dataset_gpu);
    rlt::free(device_gpu, actor_buffers_gpu);
    rlt::free(device_cpu, obs_copy);
    rlt::free(device_cpu, scalar_copy);
}
