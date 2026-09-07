#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/operations_cpu.h>
#include <rl_tools/nn_models/persist.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "../parameters.h"
#include <rl_tools/rl/environments/batch/operations_generic.h>

#include <rl_tools/rl/components/on_policy_runner/operations_cpu_mux.h>
#include <rl_tools/rl/components/on_policy_runner/operations_generic_extensions.h>

#include <rl_tools/rl/algorithms/ppo/operations_generic.h>
#include <rl_tools/rl/utils/evaluation/operations_generic.h>

#include <gtest/gtest.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>


namespace parameters = parameters_0;

using LOGGER = rlt::devices::logging::CPU_TENSORBOARD<>;

using DEV_SPEC_SUPER = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using TI = typename rlt::devices::DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints {
    struct HINTS : rlt::devices::ExecutionHints {
    };
}
struct DEV_SPEC : DEV_SPEC_SUPER {
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_GPU<rlt::devices::DefaultCUDASpecification>;


using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
using T = double;
using TI = typename DEVICE::index_t;

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, COLLECTION_CPU_GPU) {
    using penv = parameters::environment<double, TI>;
    using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
    using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = rlt::rl::components::on_policy_runner::CollectionEvaluationBuffer<prl::ON_POLICY_RUNNER_SPEC>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device_gpu);
    auto rng = rlt::random::default_engine(DEVICE{}, 12);
    auto evaluation_rng = rlt::random::default_engine(DEVICE{}, 12);
    prl::PPO_TYPE ppo;
    prl::ACTOR_OPTIMIZER actor_optimizer;
    prl::PPO_TYPE::SPEC::ACTOR_TYPE actor_gpu, actor3;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::BATCH_ENVIRONMENT environment_cpu, environment_gpu;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner_cpu, on_policy_runner_gpu;
    prl::ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer_cpu, on_policy_runner_buffer_gpu;
    prl::ON_POLICY_RUNNER_DATASET_TYPE on_policy_runner_dataset_cpu, on_policy_runner_dataset_gpu;
    ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;

    rlt::malloc(device, ppo);
    rlt::malloc(device, actor3);
    rlt::malloc(device, ppo_buffers);
    rlt::malloc(device, environment_cpu);
    rlt::malloc(device, environment_gpu);
    rlt::malloc(device, on_policy_runner_dataset_cpu);
    rlt::malloc(device, on_policy_runner_dataset_gpu);
    rlt::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
    rlt::malloc(device, on_policy_runner_cpu);
    rlt::malloc(device, on_policy_runner_gpu);
    rlt::malloc(device, on_policy_runner_buffer_cpu);
    rlt::malloc(device, on_policy_runner_buffer_gpu);
    rlt::malloc(device, actor_eval_buffers);
    rlt::malloc(device_gpu, actor_gpu);
    rlt::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
    rlt::malloc(device_gpu, actor_eval_buffers_gpu);
    rlt::init(device, environment_cpu);
    rlt::init(device, environment_gpu);
    rlt::init(device, on_policy_runner_cpu, environment_cpu, rng);
    rlt::init(device, on_policy_runner_gpu, environment_gpu, rng);

    rlt::init_weights(device, ppo.actor, rng);
    rlt::init(device, actor_optimizer);
    rlt::reset_optimizer_state(device, ppo.actor, actor_optimizer);
    rlt::reset_forward_state(device, ppo.actor);
    rlt::zero_gradient(device, ppo.actor);
    rlt::copy(device, device_gpu, ppo.actor, actor_gpu);
    rlt::copy(device_gpu, device, actor_gpu, actor3);
    auto diff = rlt::abs_diff(device, ppo.actor, actor3);
    ASSERT_LT(diff, 1e-5);

    for (TI step_i = 0; step_i < 1; step_i++) {
        auto rng_cpu_copy = rng;
        auto rng_gpu_copy = rng;
        rlt::collect(device, on_policy_runner_dataset_cpu, on_policy_runner_cpu, on_policy_runner_buffer_cpu, environment_cpu, ppo.actor, actor_eval_buffers, rng_cpu_copy);
        rlt::copy(device, device_gpu, ppo.actor, actor_gpu);
        rlt::collect_hybrid(device, device_gpu, on_policy_runner_dataset_gpu, on_policy_runner_gpu, on_policy_runner_buffer_gpu, environment_gpu, ppo.actor, actor_gpu, actor_eval_buffers_gpu, on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, rng_gpu_copy, evaluation_rng);
        for(TI rollout_step_i = 0; rollout_step_i < prl::ON_POLICY_RUNNER_STEPS_PER_ENV; rollout_step_i++) {
            auto observations_cpu_tensor = rlt::view_range(device, on_policy_runner_dataset_cpu.all_observations, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>{});
            auto observations_gpu_tensor = rlt::view_range(device, on_policy_runner_dataset_gpu.all_observations, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, rlt::tensor::ViewSpec<0, prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>{});
            auto observations_cpu = rlt::matrix_view(device, observations_cpu_tensor);
            auto observations_gpu = rlt::matrix_view(device, observations_gpu_tensor);
            auto diff_observations = rlt::abs_diff(device, observations_cpu, observations_gpu);
            auto actions_cpu = rlt::view(device, on_policy_runner_dataset_cpu.actions, rlt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::BATCH_ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto actions_gpu = rlt::view(device, on_policy_runner_dataset_gpu.actions, rlt::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::BATCH_ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto diff_actions = rlt::abs_diff(device, actions_cpu, actions_gpu);
            std::cout << "step " << step_i << " rollout_step " << rollout_step_i << " diff_observations " << diff_observations << " diff_actions " << diff_actions << std::endl;
            ASSERT_LT(diff_observations/decltype(observations_cpu)::SPEC::SIZE, 1e-5);
            ASSERT_LT(diff_actions/decltype(actions_cpu)::SPEC::SIZE, 1e-5);
        }
    }

    rlt::free(device, ppo);
    rlt::free(device, actor3);
    rlt::free(device, ppo_buffers);
    rlt::free(device, environment_cpu);
    rlt::free(device, environment_gpu);
    rlt::free(device, on_policy_runner_dataset_cpu);
    rlt::free(device, on_policy_runner_dataset_gpu);
    rlt::free(device, on_policy_runner_collection_eval_buffer_cpu);
    rlt::free(device, on_policy_runner_cpu);
    rlt::free(device, on_policy_runner_gpu);
    rlt::free(device, on_policy_runner_buffer_cpu);
    rlt::free(device, on_policy_runner_buffer_gpu);
    rlt::free(device, actor_eval_buffers);
    rlt::free(device_gpu, actor_gpu);
    rlt::free(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
    rlt::free(device_gpu, actor_eval_buffers_gpu);
}
