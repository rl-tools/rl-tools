#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_cuda.h>
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_cuda.h>
#include <rl_tools/nn/loss_functions/mse/operations_cuda.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environments/batch/operations_cuda.h>

#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>

#include <rl_tools/rl/algorithms/ppo/operations_cuda.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_cuda.h>

#include <rl_tools/rl/algorithms/ppo/operations_generic_extensions.h>
#include <rl_tools/random/operations_generic_array.h>
#include <metra/metra.h>
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
    rlt::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_SPEC> runner_buffer_gpu;
    BATCH environment_gpu;
    rlt::malloc(device_gpu, actor_gpu);
    rlt::malloc(device_gpu, actor_buffers_gpu);
    rlt::malloc(device_gpu, runner_gpu);
    rlt::malloc(device_gpu, runner_buffer_gpu);
    rlt::malloc(device_gpu, environment_gpu);

    typename DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu_init;
    rlt::malloc(device_cpu, rng_cpu_init);
    rlt::init(device_cpu, rng_cpu_init, 42);
    ACTOR_TYPE actor_cpu_init;
    rlt::malloc(device_cpu, actor_cpu_init);
    rlt::init_weights(device_cpu, actor_cpu_init, rng_cpu_init);
    rlt::copy(device_cpu, device_gpu, actor_cpu_init, actor_gpu);

    rlt::init(device_gpu, environment_gpu);
    rlt::init(device_gpu, runner_gpu, environment_gpu, rng_gpu);
    rlt::set_all(device_gpu, dataset_gpu.scalar_data, 0);
    rlt::collect(device_gpu, dataset_gpu, runner_gpu, runner_buffer_gpu, environment_gpu, actor_gpu, actor_buffers_gpu, rng_gpu);

    // Set fake values for GAE computation
    rlt::set_all(device_gpu, dataset_gpu.all_values, (T)1.0);
    rlt::set_all(device_gpu, dataset_gpu.bootstrap_values, (T)1.0);

    // Copy dataset to CPU
    DATASET dataset_cpu;
    rlt::malloc(device_cpu, dataset_cpu);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations, dataset_cpu.all_observations);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.all_observations_privileged, dataset_cpu.all_observations_privileged);
    rlt::copy(device_gpu, device_cpu, dataset_gpu.scalar_data, dataset_cpu.scalar_data);

    // Run GAE on CPU
    rlt::estimate_generalized_advantages(device_cpu, dataset_cpu, dataset_cpu.bootstrap_values, PPO_PARAMETERS{});

    // Run GAE on GPU
    rlt::estimate_generalized_advantages(device_gpu, dataset_gpu, dataset_gpu.bootstrap_values, PPO_PARAMETERS{});

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
    rlt::free(device_gpu, runner_buffer_gpu);
    rlt::free(device_gpu, environment_gpu);
    rlt::free(device_cpu, actor_cpu_init);
    rlt::free(device_cpu, adv_cpu);
    rlt::free(device_cpu, adv_gpu_copy);
}

struct PPO_IGNORE_TERMINATION: PPO_PARAMETERS{
    static constexpr bool IGNORE_TERMINATION = true;
};
template <bool IGNORE> struct PPO_NO_TRUNCATION_BOOTSTRAP: PPO_PARAMETERS{
    static constexpr bool BOOTSTRAP_TRUNCATIONS = false, IGNORE_TERMINATION = IGNORE;
};
TEST(RL_TOOLS_RL_ALGORITHMS_PPO_CUDA, GAE_BOUNDARIES){
    DEVICE_CPU cpu;
    DEVICE_GPU gpu;
    rlt::init(gpu);
    DATASET host, device;
    rlt::malloc(cpu, host); rlt::malloc(gpu, device);
    rlt::set_all(cpu, host.scalar_data, (T)0);
    rlt::set_all(cpu, host.values, (T)2);
    rlt::set_all(cpu, host.bootstrap_values, (T)5);
    rlt::set_all(cpu, host.rewards, (T)1);
    constexpr TI TIME_LIMIT = N_ENVIRONMENTS;
    constexpr TI TERMINAL = N_ENVIRONMENTS + 1;
    rlt::set(host.truncated, TIME_LIMIT, 0, (T)1);
    rlt::set(host.truncated, TERMINAL, 0, (T)1);
    rlt::set(host.terminated, TERMINAL, 0, (T)1);
    auto check = [&](auto parameters){
        rlt::copy(cpu, gpu, host.scalar_data, device.scalar_data);
        rlt::estimate_generalized_advantages(gpu, device, device.bootstrap_values, parameters);
        rlt::copy(gpu, cpu, device.target_values, host.target_values);
        EXPECT_NEAR(rlt::get(host.target_values, TIME_LIMIT, 0), parameters.BOOTSTRAP_TRUNCATIONS ? 1 + PPO_PARAMETERS::GAMMA * 5 : 1, 1e-5);
        EXPECT_NEAR(rlt::get(host.target_values, TERMINAL, 0), parameters.IGNORE_TERMINATION ? 1 + PPO_PARAMETERS::GAMMA * 5 : 1, 1e-5);
    };
    check(PPO_PARAMETERS{});
    check(PPO_IGNORE_TERMINATION{});
    check(PPO_NO_TRUNCATION_BOOTSTRAP<false>{});
    check(PPO_NO_TRUNCATION_BOOTSTRAP<true>{});
    rlt::free(cpu, host); rlt::free(gpu, device);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
    EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
}

TEST(RL_TOOLS_RL_ALGORITHMS_PPO_CUDA, HYBRID_SHAPED_OBSERVATIONS){
    using namespace rl_tools;
    struct HybridEnvironment: ENVIRONMENT{
        struct Observation: ENVIRONMENT::Observation{ using SHAPE = tensor::Shape<TI, 1, 1, 3>; };
    };
    using HybridBatch = rl::environments::batch::Independent<rl::environments::batch::Specification<HybridEnvironment, N_ENVIRONMENTS>>;
    using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<typename ActorConfig<ACTOR_CAPABILITY>::MLP_CONFIG>;
    using Chain = nn_models::sequential::Module<nn::layers::flatten::BindConfiguration<nn::layers::flatten::Configuration<TYPE_POLICY, TI>>, nn_models::sequential::Module<MLP>>;
    using Actor = nn_models::sequential::Build<ACTOR_CAPABILITY, Chain, tensor::Shape<TI, 1, N_ENVIRONMENTS, 1, 1, 3>>;
    using RS = rl::components::on_policy_runner::Specification<TYPE_POLICY, HybridBatch, typename Actor::template State<>, HybridEnvironment::Observation, HybridEnvironment::ObservationPrivileged, float, double, 3>;
    using DS = rl::components::on_policy_runner::DatasetSpecification<RS, 8>;
    using PS = rl::algorithms::ppo::Specification<TYPE_POLICY, TI, HybridEnvironment, Actor, CRITIC_TYPE, PPO_PARAMETERS>;
    using RNG = devices::generic::random::ArrayENGINE<devices::generic::random::ArraySpecification<TI, 1024>>;
    DEVICE_CPU cpu;
    DEVICE_GPU gpu;
    init(gpu);
    RNG rng_actor, rng_runner, rng_ppo, rng_evaluation;
    malloc(cpu, rng_actor); malloc(cpu, rng_runner); malloc(cpu, rng_ppo); malloc(gpu, rng_evaluation);
    init(cpu, rng_actor, 99); copy(cpu, gpu, rng_actor, rng_evaluation);
    init(cpu, rng_actor, 42); init(cpu, rng_runner, 7); init(cpu, rng_ppo, 7);
    rl::algorithms::PPO<PS> ppo;
    Actor evaluation_actor;
    typename Actor::template Buffer<> actor_buffer;
    malloc(cpu, ppo); malloc(gpu, evaluation_actor); malloc(gpu, actor_buffer);
    init_weights(cpu, ppo.actor, rng_actor); init_weights(cpu, ppo.critic, rng_actor);
    copy(cpu, gpu, ppo.actor, evaluation_actor);
    HybridBatch environment_runner, environment_ppo;
    rl::components::OnPolicyRunner<RS> runner, ppo_runner;
    rl::components::on_policy_runner::Buffer<RS> buffer, ppo_buffer;
    rl::components::on_policy_runner::Dataset<DS> dataset, ppo_dataset;
    rl::components::on_policy_runner::CollectionEvaluationBuffer<RS> transfer, ppo_transfer, evaluation_transfer;
    rl::algorithms::ppo::CollectionBuffer<CRITIC_TYPE, DS> critic_buffer;
    malloc(cpu, environment_runner); malloc(cpu, environment_ppo);
    malloc(cpu, runner); malloc(cpu, ppo_runner); malloc(cpu, buffer); malloc(cpu, ppo_buffer);
    malloc(cpu, dataset); malloc(cpu, ppo_dataset); malloc(cpu, transfer); malloc(cpu, ppo_transfer);
    malloc(gpu, evaluation_transfer); malloc(cpu, critic_buffer);
    init(cpu, environment_runner); init(cpu, environment_ppo);
    init(cpu, runner, environment_runner, rng_runner); init(cpu, ppo_runner, environment_ppo, rng_ppo);
    collect_hybrid(cpu, gpu, dataset, runner, buffer, environment_runner, ppo.actor, evaluation_actor, actor_buffer, transfer, evaluation_transfer, rng_runner, rng_evaluation);
    collect_hybrid(cpu, gpu, ppo_dataset, ppo_runner, ppo_buffer, environment_ppo, ppo, evaluation_actor, actor_buffer, ppo_transfer, evaluation_transfer, critic_buffer, rng_ppo, rng_evaluation);
    EXPECT_EQ(abs_diff(cpu, dataset.all_observations, ppo_dataset.all_observations), 0);
    EXPECT_EQ(abs_diff(cpu, dataset.all_observations_privileged, ppo_dataset.all_observations_privileged), 0);
    EXPECT_EQ(abs_diff(cpu, dataset.actions, ppo_dataset.actions), 0);
    EXPECT_EQ(abs_diff(cpu, dataset.action_log_probs, ppo_dataset.action_log_probs), 0);
    EXPECT_EQ(abs_diff(cpu, dataset.rewards, ppo_dataset.rewards), 0);
    EXPECT_EQ(abs_diff(cpu, dataset.truncated, ppo_dataset.truncated), 0);
    EXPECT_EQ(abs_diff(cpu, rng_runner, rng_ppo), 0);
    EXPECT_FALSE(is_nan(cpu, ppo_dataset.bootstrap_values));
    free(cpu, critic_buffer); free(gpu, evaluation_transfer); free(cpu, ppo_transfer); free(cpu, transfer);
    free(cpu, ppo_dataset); free(cpu, dataset); free(cpu, ppo_buffer); free(cpu, buffer);
    free(cpu, ppo_runner); free(cpu, runner); free(cpu, environment_ppo); free(cpu, environment_runner);
    free(gpu, actor_buffer); free(gpu, evaluation_actor); free(cpu, ppo);
    free(gpu, rng_evaluation); free(cpu, rng_ppo); free(cpu, rng_runner); free(cpu, rng_actor);
#ifdef RL_TOOLS_BACKEND_ENABLE_CUDNN
    EXPECT_EQ(cudnnDestroy(gpu.cudnn_handle), CUDNN_STATUS_SUCCESS);
#endif
    EXPECT_EQ(cublasDestroy(gpu.handle), CUBLAS_STATUS_SUCCESS);
    EXPECT_EQ(cudaStreamDestroy(gpu.stream), cudaSuccess);
    metra::log("ppo/hybrid/failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
}
