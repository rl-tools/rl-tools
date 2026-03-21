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

#include <rl_tools/rl/components/on_policy_runner/operations_cpu.h>
#include <rl_tools/rl/components/on_policy_runner/operations_cuda.h>

#include <rl_tools/rl/algorithms/ppo/operations_cuda.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_cuda.h>

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
constexpr TI BATCH_SIZE = N_ENVIRONMENTS * STEPS_PER_ENV;

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

using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<TYPE_POLICY, TI, ENVIRONMENT, ACTOR_TYPE::State<>, N_ENVIRONMENTS, ENVIRONMENT::EPISODE_STEP_LIMIT>;
using DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, STEPS_PER_ENV>;
using DATASET = rlt::rl::components::on_policy_runner::Dataset<DATASET_SPEC>;

using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<rlt::devices::random::CUDA::Specification<TI, 1024>>;

struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
    static constexpr TI N_EPOCHS = 1;
    static constexpr bool LEARN_ACTION_STD = true;
    static constexpr T INITIAL_ACTION_STD = 0.5;
    static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.01;
    static constexpr bool NORMALIZE_ADVANTAGE = true;
    static constexpr T GAMMA = 0.99;
};

using CRITIC_CONFIG = rlt::nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY>;
using CRITIC_MODULE = rlt::nn_models::mlp_unconditional_stddev::BindConfiguration<CRITIC_CONFIG>;
using CRITIC_CHAIN = rlt::nn_models::sequential::Module<CRITIC_MODULE>;
using CRITIC_TYPE = rlt::nn_models::sequential::Build<ACTOR_CAPABILITY, CRITIC_CHAIN, rlt::tensor::Shape<TI, 1, BATCH_SIZE, ENVIRONMENT::Observation::DIM>>;

using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<TYPE_POLICY, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
using PPO_TYPE = rlt::rl::algorithms::PPO<PPO_SPEC>;
using PPO_BUFFERS_SPEC = rlt::rl::algorithms::ppo::BufferSpecification<PPO_SPEC>;
using PPO_BUFFERS = rlt::rl::algorithms::ppo::Buffers<PPO_BUFFERS_SPEC>;


TEST(RL_TOOLS_RL_ALGORITHMS_PPO_CUDA, GAE){
    DEVICE_CPU device_cpu;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);

    // Create dataset on GPU, fill with known values, run GAE on both
    DATASET dataset_gpu;
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, dataset_gpu);
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, 42);

    // Fill dataset with synthetic data by running collect on GPU
    ACTOR_TYPE actor_gpu;
    typename ACTOR_TYPE::template CHANGE_BATCH_SIZE<TI, N_ENVIRONMENTS>::template Buffer<> actor_buffers_gpu;
    rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC> runner_gpu;
    rlt::malloc(device_gpu, actor_gpu);
    rlt::malloc(device_gpu, actor_buffers_gpu);
    rlt::malloc(device_gpu, runner_gpu);

    typename DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu_init;
    rlt::malloc(device_cpu, rng_cpu_init);
    rlt::init(device_cpu, rng_cpu_init, 42);
    ACTOR_TYPE actor_cpu_init;
    rlt::malloc(device_cpu, actor_cpu_init);
    rlt::init_weights(device_cpu, actor_cpu_init, rng_cpu_init);
    rlt::copy(device_cpu, device_gpu, actor_cpu_init, actor_gpu);

    rlt::Tensor<rlt::tensor::Specification<ENVIRONMENT, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS>>> envs;
    rlt::Tensor<rlt::tensor::Specification<ENVIRONMENT::Parameters, TI, rlt::tensor::Shape<TI, N_ENVIRONMENTS>>> params;
    rlt::init(device_gpu, runner_gpu, envs, params, actor_gpu, rng_gpu);
    rlt::set_all(device_gpu, dataset_gpu.scalar_data, 0);
    rlt::collect(device_gpu, dataset_gpu, runner_gpu, actor_gpu, actor_buffers_gpu, rng_gpu);

    // Set fake values for GAE computation
    rlt::set_all(device_gpu, dataset_gpu.all_values, (T)1.0);

    // Copy dataset to CPU
    DATASET dataset_cpu;
    rlt::malloc(device_cpu, dataset_cpu);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations, dataset_cpu.all_observations);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations_privileged, dataset_cpu.all_observations_privileged);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.scalar_data, dataset_cpu.scalar_data);

    // Run GAE on CPU
    rlt::estimate_generalized_advantages(device_cpu, dataset_cpu, PPO_PARAMETERS{});

    // Run GAE on GPU
    rlt::estimate_generalized_advantages(device_gpu, dataset_gpu, PPO_PARAMETERS{});

    // Copy GPU advantages back
    rlt::Matrix<rlt::matrix::Specification<T, TI, DATASET_SPEC::STEPS_TOTAL, 1>> adv_cpu, adv_gpu_copy;
    rlt::malloc(device_cpu, adv_cpu);
    rlt::malloc(device_cpu, adv_gpu_copy);

    // Copy from datasets
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL; i++){
        rlt::set(adv_cpu, i, 0, rlt::get(dataset_cpu.advantages, i, 0));
    }
    rlt::copy(device_gpu, device_cpu, dataset_gpu.advantages, adv_gpu_copy);

    // Compare
    T diff = 0;
    for(TI i = 0; i < DATASET_SPEC::STEPS_TOTAL; i++){
        T cpu_val = rlt::get(adv_cpu, i, 0);
        T gpu_val = rlt::get(adv_gpu_copy, i, 0);
        diff += rlt::math::abs(device_cpu.math, cpu_val - gpu_val);
    }
    std::cout << "GAE abs_diff: " << diff << std::endl;
    EXPECT_LT(diff, 1e-2) << "GAE CPU vs GPU should match closely (float precision allows ~1e-6 per element)";

    // Cleanup
    rlt::free(device_cpu, dataset_cpu);
    rlt::free(device_gpu, dataset_gpu);
    rlt::free(device_gpu, actor_gpu);
    rlt::free(device_gpu, actor_buffers_gpu);
    rlt::free(device_gpu, runner_gpu);
    rlt::free(device_cpu, actor_cpu_init);
    rlt::free(device_cpu, adv_cpu);
    rlt::free(device_cpu, adv_gpu_copy);
}
