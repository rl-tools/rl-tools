#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_PERSIST_H
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        save(device, dataset.data, group, "data");
        save(device, dataset.all_observations_privileged, group, "all_observations");
        save(device, dataset.observations, group, "observations");
        save(device, dataset.actions, group, "actions");
        save(device, dataset.action_log_probs, group, "action_log_probs");
        save(device, dataset.rewards, group, "rewards");
        save(device, dataset.terminated, group, "terminated");
        save(device, dataset.truncated, group, "truncated");
        save(device, dataset.all_values, group, "all_values");
        save(device, dataset.values, group, "values");
        save(device, dataset.advantages, group, "advantages");
        save(device, dataset.target_values, group, "target_values");
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::on_policy_runner::Dataset<SPEC>& dataset, GROUP& group){
        bool success = load(device, dataset.data, group, "data");
        success &= load(device, dataset.all_observations_privileged, group, "all_observations");
        success &= load(device, dataset.observations, group, "observations");
        success &= load(device, dataset.actions, group, "actions");
        success &= load(device, dataset.action_log_probs, group, "action_log_probs");
        success &= load(device, dataset.rewards, group, "rewards");
        success &= load(device, dataset.terminated, group, "terminated");
        success &= load(device, dataset.truncated, group, "truncated");
        success &= load(device, dataset.all_values, group, "all_values");
        success &= load(device, dataset.values, group, "values");
        success &= load(device, dataset.advantages, group, "advantages");
        success &= load(device, dataset.target_values, group, "target_values");
        return success;
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        set(device, step_tensor, runner.step, 0);
        save(device, step_tensor, group, "step");
        free(device, step_tensor);
        auto policy_state_group = create_group(device, group, "policy_state");
        save(device, runner.policy_state, policy_state_group);
        static constexpr TI STATE_SIZE_BYTES = SPEC::N_ENVIRONMENTS * sizeof(typename SPEC::ENVIRONMENT::State);
        static constexpr TI STATE_SIZE_T = (STATE_SIZE_BYTES + sizeof(T) - 1) / sizeof(T);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, STATE_SIZE_T>>> states_raw;
        malloc(device, states_raw);
        std::memcpy(data(states_raw), &get(runner.states, 0, 0), STATE_SIZE_BYTES);
        save(device, states_raw, group, "states");
        free(device, states_raw);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> truncated_tensor;
        malloc(device, truncated_tensor);
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(device, truncated_tensor, get(runner.truncated, 0, i) ? (T)1 : (T)0, i);
        }
        save(device, truncated_tensor, group, "truncated");
        free(device, truncated_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> episode_step_tensor;
        malloc(device, episode_step_tensor);
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(device, episode_step_tensor, get(runner.episode_step, 0, i), i);
        }
        save(device, episode_step_tensor, group, "episode_step");
        free(device, episode_step_tensor);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> episode_return_tensor;
        malloc(device, episode_return_tensor);
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(device, episode_return_tensor, get(runner.episode_return, 0, i), i);
        }
        save(device, episode_return_tensor, group, "episode_return");
        free(device, episode_return_tensor);
    }
    template <typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, rl::components::OnPolicyRunner<SPEC>& runner, GROUP& group){
        using TI = typename DEVICE::index_t;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        bool success = load(device, step_tensor, group, "step");
        runner.step = get(device, step_tensor, 0);
        free(device, step_tensor);
        auto policy_state_group = get_group(device, group, "policy_state");
        success &= load(device, runner.policy_state, policy_state_group);
        static constexpr TI STATE_SIZE_BYTES = SPEC::N_ENVIRONMENTS * sizeof(typename SPEC::ENVIRONMENT::State);
        static constexpr TI STATE_SIZE_T = (STATE_SIZE_BYTES + sizeof(T) - 1) / sizeof(T);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, STATE_SIZE_T>>> states_raw;
        malloc(device, states_raw);
        success &= load(device, states_raw, group, "states");
        std::memcpy(&get(runner.states, 0, 0), data(states_raw), STATE_SIZE_BYTES);
        free(device, states_raw);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> truncated_tensor;
        malloc(device, truncated_tensor);
        success &= load(device, truncated_tensor, group, "truncated");
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(runner.truncated, 0, i, get(device, truncated_tensor, i) > (T)0.5);
        }
        free(device, truncated_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> episode_step_tensor;
        malloc(device, episode_step_tensor);
        success &= load(device, episode_step_tensor, group, "episode_step");
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(runner.episode_step, 0, i, get(device, episode_step_tensor, i));
        }
        free(device, episode_step_tensor);
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, SPEC::N_ENVIRONMENTS>>> episode_return_tensor;
        malloc(device, episode_return_tensor);
        success &= load(device, episode_return_tensor, group, "episode_return");
        for(TI i = 0; i < SPEC::N_ENVIRONMENTS; i++){
            set(runner.episode_return, 0, i, get(device, episode_return_tensor, i));
        }
        free(device, episode_return_tensor);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
