#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_PERSIST_H
#include "state.h"
#include "../../../../../nn/optimizers/adam/persist.h"
#include "../../../../../nn/optimizers/adam/instance/persist.h"
#include "../../../../../nn_models/mlp_unconditional_stddev/persist.h"  // Must be before sequential for correct overload resolution
#include "../../../../../nn_models/sequential/persist.h"
#include "../../../../../rl/algorithms/ppo/persist.h"
#include "../../../../../rl/components/on_policy_runner/persist.h"
#include "../../../../../rl/components/running_normalizer/persist.h"
#include "../../../../../random/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    void save(DEVICE& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts, GROUP& group){
        using TI = typename DEVICE::index_t;
        using T = typename T_CONFIG::T;
        auto actor_optimizer_group = create_group(device, group, "actor_optimizer");
        save(device, ts.actor_optimizer, actor_optimizer_group);
        auto critic_optimizer_group = create_group(device, group, "critic_optimizer");
        save(device, ts.critic_optimizer, critic_optimizer_group);
        auto ppo_group = create_group(device, group, "ppo");
        save(device, ts.ppo, ppo_group);
        auto on_policy_runner_group = create_group(device, group, "on_policy_runner");
        save(device, ts.on_policy_runner, on_policy_runner_group);
        auto on_policy_runner_dataset_group = create_group(device, group, "on_policy_runner_dataset");
        save(device, ts.on_policy_runner_dataset, on_policy_runner_dataset_group);
        auto observation_normalizer_group = create_group(device, group, "observation_normalizer");
        save(device, ts.observation_normalizer, observation_normalizer_group);
        auto observation_privileged_normalizer_group = create_group(device, group, "observation_privileged_normalizer");
        save(device, ts.observation_privileged_normalizer, observation_privileged_normalizer_group);
        auto rng_group = create_group(device, group, "rng");
        save(device, ts.rng, rng_group);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        set(device, step_tensor, ts.step, 0);
        save(device, step_tensor, group, "step");
        free(device, step_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> next_checkpoint_id_tensor;
        malloc(device, next_checkpoint_id_tensor);
        set(device, next_checkpoint_id_tensor, ts.next_checkpoint_id, 0);
        save(device, next_checkpoint_id_tensor, group, "next_checkpoint_id");
        free(device, next_checkpoint_id_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> next_evaluation_id_tensor;
        malloc(device, next_evaluation_id_tensor);
        set(device, next_evaluation_id_tensor, ts.next_evaluation_id, 0);
        save(device, next_evaluation_id_tensor, group, "next_evaluation_id");
        free(device, next_evaluation_id_tensor);
    }
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    bool load(DEVICE& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts, GROUP& group){
        using TI = typename DEVICE::index_t;
        using T = typename T_CONFIG::T;
        bool success = true;
        bool step_result;
        auto actor_optimizer_group = get_group(device, group, "actor_optimizer");
        step_result = load(device, ts.actor_optimizer, actor_optimizer_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: actor_optimizer"); }
        success &= step_result;
        auto critic_optimizer_group = get_group(device, group, "critic_optimizer");
        step_result = load(device, ts.critic_optimizer, critic_optimizer_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: critic_optimizer"); }
        success &= step_result;
        auto ppo_group = get_group(device, group, "ppo");
        step_result = load(device, ts.ppo, ppo_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: ppo"); }
        success &= step_result;
        auto on_policy_runner_group = get_group(device, group, "on_policy_runner");
        step_result = load(device, ts.on_policy_runner, on_policy_runner_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: on_policy_runner"); }
        success &= step_result;
        auto on_policy_runner_dataset_group = get_group(device, group, "on_policy_runner_dataset");
        step_result = load(device, ts.on_policy_runner_dataset, on_policy_runner_dataset_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: on_policy_runner_dataset"); }
        success &= step_result;
        auto observation_normalizer_group = get_group(device, group, "observation_normalizer");
        step_result = load(device, ts.observation_normalizer, observation_normalizer_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: observation_normalizer"); }
        success &= step_result;
        auto observation_privileged_normalizer_group = get_group(device, group, "observation_privileged_normalizer");
        step_result = load(device, ts.observation_privileged_normalizer, observation_privileged_normalizer_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: observation_privileged_normalizer"); }
        success &= step_result;
        auto rng_group = get_group(device, group, "rng");
        step_result = load(device, ts.rng, rng_group);
        if(!step_result){ log(device, device.logger, "PPO loop load failed: rng"); }
        success &= step_result;
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> step_tensor;
        malloc(device, step_tensor);
        step_result = load(device, step_tensor, group, "step");
        if(!step_result){ log(device, device.logger, "PPO loop load failed: step"); }
        success &= step_result;
        ts.step = get(device, step_tensor, 0);
        free(device, step_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> next_checkpoint_id_tensor;
        malloc(device, next_checkpoint_id_tensor);
        step_result = load(device, next_checkpoint_id_tensor, group, "next_checkpoint_id");
        if(!step_result){ log(device, device.logger, "PPO loop load failed: next_checkpoint_id"); }
        success &= step_result;
        ts.next_checkpoint_id = get(device, next_checkpoint_id_tensor, 0);
        free(device, next_checkpoint_id_tensor);
        Tensor<tensor::Specification<TI, TI, tensor::Shape<TI, 1>>> next_evaluation_id_tensor;
        malloc(device, next_evaluation_id_tensor);
        step_result = load(device, next_evaluation_id_tensor, group, "next_evaluation_id");
        if(!step_result){ log(device, device.logger, "PPO loop load failed: next_evaluation_id"); }
        success &= step_result;
        ts.next_evaluation_id = get(device, next_evaluation_id_tensor, 0);
        free(device, next_evaluation_id_tensor);
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
