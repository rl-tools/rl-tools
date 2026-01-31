#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        // All fields (observations, actions, rewards, etc.) are views into dataset.data
        // So saving dataset.data is sufficient
        save(device, dataset.data, group, "data");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        // All fields are views into dataset.data, so loading data restores everything
        return load(device, dataset.data, group, "data");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        set(device, step_tensor, runner.step, 0);
        save(device, step_tensor, group, "step");
        free(device, step_tensor);
        auto policy_state_group = create_group(device, group, "policy_state");
        save(device, runner.policy_state, policy_state_group);
        save_binary(device, &get(runner.states, 0, 0), SPEC::N_ENVIRONMENTS, group, "states");
        save_binary(device, &get(runner.env_parameters, 0, 0), SPEC::N_ENVIRONMENTS, group, "env_parameters");
        save_binary(device, &get(runner.environments, 0, 0), SPEC::N_ENVIRONMENTS, group, "environments");
        save(device, runner.truncated, group, "truncated");
        save(device, runner.episode_step, group, "episode_step");
        save(device, runner.episode_return, group, "episode_return");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        bool success = load(device, step_tensor, group, "step");
        runner.step = get(device, step_tensor, 0);
        free(device, step_tensor);
        auto policy_state_group = get_group(device, group, "policy_state");
        success &= load(device, runner.policy_state, policy_state_group);
        success &= load_binary(device, &get(runner.states, 0, 0), SPEC::N_ENVIRONMENTS, group, "states");
        success &= load_binary(device, &get(runner.env_parameters, 0, 0), SPEC::N_ENVIRONMENTS, group, "env_parameters");
        success &= load_binary(device, &get(runner.environments, 0, 0), SPEC::N_ENVIRONMENTS, group, "environments");
        success &= load(device, runner.truncated, group, "truncated");
        success &= load(device, runner.episode_step, group, "episode_step");
        success &= load(device, runner.episode_return, group, "episode_return");
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
