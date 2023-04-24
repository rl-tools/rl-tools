#define BACKPROP_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA

#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>
#include <backprop_tools/nn_models/operations_cpu.h>
#include <backprop_tools/nn_models/persist.h>

namespace bpt = backprop_tools;

#include "../parameters_ppo.h"

#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_MKL

#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_mkl.h>
#include <backprop_tools/rl/components/on_policy_runner/operations_generic_extensions.h>

#else
#ifdef BACKPROP_TOOLS_BACKEND_ENABLE_ACCELERATE
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <backprop_tools/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif

#include <backprop_tools/rl/algorithms/ppo/operations_generic.h>
#include <backprop_tools/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = bpt::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = bpt::devices::cpu::Specification<bpt::devices::math::CPU, bpt::devices::random::CPU, LOGGER>;
using TI = typename bpt::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints {
    struct HINTS : bpt::rl::components::on_policy_runner::ExecutionHints<TI, 16> {
    };
}
struct DEV_SPEC : DEV_SPEC_SUPER {
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = bpt::DEVICE_FACTORY_GPU<bpt::devices::DefaultCUDASpecification>;


using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
using T = double;
using TI = typename DEVICE::index_t;

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, COLLECTION_CPU_GPU) {
    using penv = parameters::environment<double, TI>;
    using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
    using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = bpt::rl::components::on_policy_runner::CollectionEvaluationBuffer<prl::ON_POLICY_RUNNER_SPEC>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_GPU device_gpu;
    bpt::init(device_gpu);
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    auto evaluation_rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    prl::PPO_TYPE ppo;
    prl::ACTOR_OPTIMIZER actor_optimizer;
    prl::PPO_TYPE::SPEC::ACTOR_TYPE actor_gpu, actor3;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner_cpu, on_policy_runner_gpu;
    prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset_cpu, on_policy_runner_dataset_gpu;
    ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;
    penv::ENVIRONMENT envs_cpu[prl::N_ENVIRONMENTS];
    penv::ENVIRONMENT envs_gpu[prl::N_ENVIRONMENTS];

    bpt::malloc(device, ppo);
    bpt::malloc(device, actor3);
    bpt::malloc(device, ppo_buffers);
    bpt::malloc(device, on_policy_runner_dataset_cpu);
    bpt::malloc(device, on_policy_runner_dataset_gpu);
    bpt::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
    bpt::malloc(device, on_policy_runner_cpu);
    bpt::malloc(device, on_policy_runner_gpu);
    bpt::malloc(device, actor_eval_buffers);
    bpt::malloc(device_gpu, actor_gpu);
    bpt::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
    bpt::malloc(device_gpu, actor_eval_buffers_gpu);
    for (auto &env: envs_cpu) {
        bpt::malloc(device, env);
    }
    for (auto &env: envs_gpu) {
        bpt::malloc(device, env);
    }

    bpt::init(device, on_policy_runner_cpu, envs_cpu, rng);
    bpt::init(device, on_policy_runner_gpu, envs_gpu, rng);

    bpt::init_weights(device, ppo.actor, rng);
    bpt::reset_optimizer_state(device, ppo.actor, actor_optimizer);
    bpt::reset_forward_state(device, ppo.actor);
    bpt::zero_gradient(device, ppo.actor);
    bpt::copy(device_gpu, device, actor_gpu, ppo.actor);
    bpt::copy(device, device_gpu, actor3, actor_gpu);
    auto diff = bpt::abs_diff(device, ppo.actor, actor3);
    ASSERT_LT(diff, 1e-5);

    for (TI step_i = 0; step_i < 1; step_i++) {
        auto rng_cpu_copy = rng;
        auto rng_gpu_copy = rng;
        bpt::collect(device, on_policy_runner_dataset_cpu, on_policy_runner_cpu, ppo.actor, actor_eval_buffers, rng_cpu_copy);
        bpt::copy(device_gpu, device, actor_gpu, ppo.actor);
        bpt::collect_hybrid(device, device_gpu, on_policy_runner_dataset_gpu, on_policy_runner_gpu, ppo.actor, actor_gpu, actor_eval_buffers_gpu, on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, rng_gpu_copy);
        for(TI rollout_step_i = 0; rollout_step_i < prl::ON_POLICY_RUNNER_STEPS_PER_ENV; rollout_step_i++) {
            auto observations_cpu = bpt::view(device, on_policy_runner_dataset_cpu.observations, bpt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::OBSERVATION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto observations_gpu = bpt::view(device, on_policy_runner_dataset_gpu.observations, bpt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::OBSERVATION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto diff_observations = bpt::abs_diff(device, observations_cpu, observations_gpu);
            auto actions_cpu = bpt::view(device, on_policy_runner_dataset_cpu.actions, bpt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto actions_gpu = bpt::view(device, on_policy_runner_dataset_gpu.actions, bpt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto diff_actions = bpt::abs_diff(device, actions_cpu, actions_gpu);
            std::cout << "step " << step_i << " rollout_step " << rollout_step_i << " diff_observations " << diff_observations << " diff_actions " << diff_actions << std::endl;
            ASSERT_LT(diff_observations/decltype(observations_cpu)::SPEC::SIZE, 1e-5);
            ASSERT_LT(diff_actions/decltype(actions_cpu)::SPEC::SIZE, 1e-5);
        }
    }
}
