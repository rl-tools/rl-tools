#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H
#include "on_policy_runner.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        save(device, dataset.all_observations, group, "all_observations");
        save(device, dataset.all_observations_privileged, group, "all_observations_privileged");
        save_binary(device, &get_ref(device, dataset.episode_end_reason, 0, 0), (SPEC::STEPS_PER_ENV + 1) * SPEC::SPEC::N_ENVIRONMENTS, group, "episode_end_reason");
        save(device, dataset.episode_length, group, "episode_length");
        save(device, dataset.episode_return, group, "episode_return");
        save(device, dataset.scalar_data, group, "data");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        bool success = load(device, dataset.all_observations, group, "all_observations");
        success &= load(device, dataset.all_observations_privileged, group, "all_observations_privileged");
        success &= load_binary(device, &get_ref(device, dataset.episode_end_reason, 0, 0), (SPEC::STEPS_PER_ENV + 1) * SPEC::SPEC::N_ENVIRONMENTS, group, "episode_end_reason");
        success &= load(device, dataset.episode_length, group, "episode_length");
        success &= load(device, dataset.episode_return, group, "episode_return");
        success &= load(device, dataset.scalar_data, group, "data");
        return success;
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        set(device, step_tensor, runner.step, 0);
        save(device, step_tensor, group, "step");
        auto policy_state_group = create_group(device, group, "policy_state");
        save(device, runner.policy_state, policy_state_group);
        save_binary(device, &get_ref(device, runner.states, 0), SPEC::N_ENVIRONMENTS, group, "states");
        save_binary(device, &get_ref(device, runner.env_parameters, 0), SPEC::N_ENVIRONMENTS, group, "env_parameters");
        auto episodes_group = create_group(device, group, "episodes");
        save(device, runner.episode_step, episodes_group, "episode_step");
        save(device, runner.reset, episodes_group, "reset");
        save(device, runner.episode_return, episodes_group, "episode_return");
        save(device, runner.completed_episode_length, episodes_group, "completed_episode_length");
        save(device, runner.completed_episode_return, episodes_group, "completed_episode_return");
        save_binary(device, &get_ref(device, runner.completed_episode_reason, 0), SPEC::N_ENVIRONMENTS, episodes_group, "completed_episode_reason");
        set(device, step_tensor, runner.episode_step_limit, 0);
        save(device, step_tensor, episodes_group, "step_limit");
        free(device, step_tensor);
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        bool success = load(device, step_tensor, group, "step");
        runner.step = get(device, step_tensor, 0);
        auto policy_state_group = get_group(device, group, "policy_state");
        success &= load(device, runner.policy_state, policy_state_group);
        success &= load_binary(device, &get_ref(device, runner.states, 0), SPEC::N_ENVIRONMENTS, group, "states");
        success &= load_binary(device, &get_ref(device, runner.env_parameters, 0), SPEC::N_ENVIRONMENTS, group, "env_parameters");
        auto episodes_group = get_group(device, group, "episodes");
        success &= load(device, runner.episode_step, episodes_group, "episode_step");
        success &= load(device, runner.reset, episodes_group, "reset");
        success &= load(device, runner.episode_return, episodes_group, "episode_return");
        success &= load(device, runner.completed_episode_length, episodes_group, "completed_episode_length");
        success &= load(device, runner.completed_episode_return, episodes_group, "completed_episode_return");
        success &= load_binary(device, &get_ref(device, runner.completed_episode_reason, 0), SPEC::N_ENVIRONMENTS, episodes_group, "completed_episode_reason");
        success &= load(device, step_tensor, episodes_group, "step_limit");
        runner.episode_step_limit = get(device, step_tensor, 0);
        free(device, step_tensor);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
