#define LAYER_IN_C_OPERATIONS_CPU_MUX_INCLUDE_CUDA

#include <layer_in_c/operations/cpu_mux.h>
#include <layer_in_c/nn/operations_cpu_mux.h>
#include <layer_in_c/nn_models/operations_cpu.h>
#include <layer_in_c/nn_models/persist.h>

namespace lic = layer_in_c;

#include "../parameters_ppo.h"

#ifdef LAYER_IN_C_BACKEND_ENABLE_MKL

#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_mkl.h>
#include <layer_in_c/rl/components/on_policy_runner/operations_generic_extensions.h>

#else
#ifdef LAYER_IN_C_BACKEND_ENABLE_ACCELERATE
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu_accelerate.h>
#else
#include <layer_in_c/rl/components/on_policy_runner/operations_cpu.h>
#endif
#endif

#include <layer_in_c/rl/algorithms/ppo/operations_generic.h>
#include <layer_in_c/rl/utils/evaluation.h>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>


namespace parameters = parameters_0;

using LOGGER = lic::devices::logging::CPU_TENSORBOARD;

using DEV_SPEC_SUPER = lic::devices::cpu::Specification<lic::devices::math::CPU, lic::devices::random::CPU, LOGGER>;
using TI = typename DEVICE_FACTORY<DEV_SPEC_SUPER>::index_t;
namespace execution_hints {
    struct HINTS : lic::rl::components::on_policy_runner::ExecutionHints<TI, 16> {
    };
}
struct DEV_SPEC : DEV_SPEC_SUPER {
    using EXECUTION_HINTS = execution_hints::HINTS;
};
using DEVICE = DEVICE_FACTORY<DEV_SPEC>;
using DEVICE_GPU = DEVICE_FACTORY_GPU<lic::devices::DefaultCUDASpecification>;


using DEVICE = DEVICE_FACTORY<DEV_SPEC>;
using T = double;
using TI = typename DEVICE::index_t;

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, COLLECTION_CPU_GPU) {
    using penv = parameters::environment<double, TI>;
    using prl = parameters::rl<T, TI, penv::ENVIRONMENT>;
    using ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE = lic::rl::components::on_policy_runner::CollectionEvaluationBuffer<prl::ON_POLICY_RUNNER_SPEC>;

    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    DEVICE_GPU device_gpu;
    lic::init(device_gpu);
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    auto evaluation_rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 12);
    prl::PPO_TYPE ppo;
    prl::OPTIMIZER optimizer;
    prl::PPO_TYPE::SPEC::ACTOR_TYPE actor_gpu, actor3;
    prl::PPO_BUFFERS_TYPE ppo_buffers;
    prl::ON_POLICY_RUNNER_TYPE on_policy_runner_cpu, on_policy_runner_gpu;
    prl::ON_POLICY_RUNNER_BUFFER_TYPE on_policy_runner_buffer_cpu, on_policy_runner_buffer_gpu;
    ON_POLICY_RUNNER_COLLECTION_EVALUATION_BUFFER_TYPE on_policy_runner_collection_eval_buffer_gpu, on_policy_runner_collection_eval_buffer_cpu;
    prl::ACTOR_EVAL_BUFFERS actor_eval_buffers, actor_eval_buffers_gpu;
    penv::ENVIRONMENT envs_cpu[prl::N_ENVIRONMENTS];
    penv::ENVIRONMENT envs_gpu[prl::N_ENVIRONMENTS];

    lic::malloc(device, ppo);
    lic::malloc(device, actor3);
    lic::malloc(device, ppo_buffers);
    lic::malloc(device, on_policy_runner_buffer_cpu);
    lic::malloc(device, on_policy_runner_buffer_gpu);
    lic::malloc(device, on_policy_runner_collection_eval_buffer_cpu);
    lic::malloc(device, on_policy_runner_cpu);
    lic::malloc(device, on_policy_runner_gpu);
    lic::malloc(device, actor_eval_buffers);
    lic::malloc(device_gpu, actor_gpu);
    lic::malloc(device_gpu, on_policy_runner_collection_eval_buffer_gpu);
    lic::malloc(device_gpu, actor_eval_buffers_gpu);
    for (auto &env: envs_cpu) {
        lic::malloc(device, env);
    }
    for (auto &env: envs_gpu) {
        lic::malloc(device, env);
    }

    lic::init(device, on_policy_runner_cpu, envs_cpu, rng);
    lic::init(device, on_policy_runner_gpu, envs_gpu, rng);

    lic::init_weights(device, ppo.actor, rng);
    lic::reset_optimizer_state(device, ppo.actor, optimizer);
    lic::reset_forward_state(device, ppo.actor);
    lic::zero_gradient(device, ppo.actor);
    lic::copy(device_gpu, device, actor_gpu, ppo.actor);
    lic::copy(device, device_gpu, actor3, actor_gpu);
    auto diff = lic::abs_diff(device, ppo.actor, actor3);
    ASSERT_LT(diff, 1e-5);

    for (TI step_i = 0; step_i < 1; step_i++) {
        auto rng_cpu_copy = rng;
        auto rng_gpu_copy = rng;
        lic::collect(device, on_policy_runner_buffer_cpu, on_policy_runner_cpu, ppo.actor, actor_eval_buffers, rng_cpu_copy);
        lic::copy(device_gpu, device, actor_gpu, ppo.actor);
        lic::collect_hybrid(device, device_gpu, on_policy_runner_buffer_gpu, on_policy_runner_gpu, ppo.actor, actor_gpu, actor_eval_buffers_gpu, on_policy_runner_collection_eval_buffer_cpu, on_policy_runner_collection_eval_buffer_gpu, rng_gpu_copy);
        for(TI rollout_step_i = 0; rollout_step_i < prl::ON_POLICY_RUNNER_STEPS_PER_ENV; rollout_step_i++) {
            auto observations_cpu = lic::view(device, on_policy_runner_buffer_cpu.observations, lic::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::OBSERVATION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto observations_gpu = lic::view(device, on_policy_runner_buffer_gpu.observations, lic::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::OBSERVATION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto diff_observations = lic::abs_diff(device, observations_cpu, observations_gpu);
            auto actions_cpu = lic::view(device, on_policy_runner_buffer_cpu.actions, lic::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto actions_gpu = lic::view(device, on_policy_runner_buffer_gpu.actions, lic::matrix::ViewSpec<prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, prl::ON_POLICY_RUNNER_SPEC::ENVIRONMENT::ACTION_DIM>{}, rollout_step_i * prl::ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS, 0);
            auto diff_actions = lic::abs_diff(device, actions_cpu, actions_gpu);
            std::cout << "step " << step_i << " rollout_step " << rollout_step_i << " diff_observations " << diff_observations << " diff_actions " << diff_actions << std::endl;
            ASSERT_LT(diff_observations/decltype(observations_cpu)::SPEC::SIZE, 1e-5);
            ASSERT_LT(diff_actions/decltype(actions_cpu)::SPEC::SIZE, 1e-5);
        }
    }
}
