#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_GENERIC_H

#include "../../../../../nn/optimizers/adam/instance/operations_generic.h"
#include "../../../../../nn/layers/standardize/operations_generic.h"
#include "../../../../../nn_models/mlp_unconditional_stddev/operations_generic.h"
#include "../../../../../nn_models/sequential/operations_generic.h"
#include "../../../../../nn/optimizers/adam/operations_generic.h"
#include "../../../../../rl/algorithms/ppo/operations_generic.h"
#include "../../../../../rl/components/on_policy_runner/operations_generic.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    // Device-dependent helpers (overridden by operations_cuda.h for CUDA)
    template <typename DEVICE, typename ENV_SPEC>
    void malloc_environments(DEVICE& device, Tensor<ENV_SPEC>& envs){
        using TI = typename DEVICE::index_t;
        constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
        for(TI env_i = 0; env_i < N; env_i++){
            auto& env = get_ref(device, envs, env_i);
            malloc(device, env);
        }
    }
    template <typename DEVICE, typename ENV_SPEC>
    void free_environments(DEVICE& device, Tensor<ENV_SPEC>& envs){
        using TI = typename DEVICE::index_t;
        constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
        for(TI env_i = 0; env_i < N; env_i++){
            auto& env = get_ref(device, envs, env_i);
            free(device, env);
        }
    }
    template <typename DEVICE, typename ENV_SPEC>
    void init_environments(DEVICE& device, Tensor<ENV_SPEC>& envs){
        using TI = typename DEVICE::index_t;
        constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
        for(TI env_i = 0; env_i < N; env_i++){
            auto& env = get_ref(device, envs, env_i);
            init(device, env);
        }
    }
    // Shared malloc/free/init
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts){
        malloc(device, ts.rng);
        malloc(device, ts.ppo);
        malloc(device, ts.ppo_buffers);
        malloc(device, ts.on_policy_runner_dataset);
        malloc(device, ts.on_policy_runner);
        malloc(device, ts.actor_eval_buffers);
        malloc(device, ts.actor_buffers);
        malloc(device, ts.critic_buffers);
        malloc(device, ts.critic_buffers_gae);
        malloc(device, ts.actor_optimizer);
        malloc(device, ts.critic_optimizer);
        malloc(device, ts.envs);
        malloc(device, ts.env_parameters);
        malloc_environments(device, ts.envs);
    }
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts){
        free(device, ts.rng);
        free(device, ts.ppo);
        free(device, ts.ppo_buffers);
        free(device, ts.on_policy_runner_dataset);
        free(device, ts.on_policy_runner);
        free(device, ts.actor_eval_buffers);
        free(device, ts.actor_buffers);
        free(device, ts.critic_buffers);
        free(device, ts.critic_buffers_gae);
        free(device, ts.actor_optimizer);
        free(device, ts.critic_optimizer);
        free(device, ts.envs);
        free(device, ts.env_parameters);
        free_environments(device, ts.envs);
    }
    template <typename DEVICE, typename T_CONFIG>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using CONFIG = T_CONFIG;
        using TI = typename DEVICE::index_t;

        init(device, ts.rng, seed);
        init_environments(device, ts.envs);
        init(device, ts.ppo, ts.actor_optimizer, ts.critic_optimizer, ts.rng);
        init(device, ts.on_policy_runner, ts.envs, ts.env_parameters, ts.ppo.actor, ts.rng);

        ts.step = 0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "operations_generic_per_loop_step.h"

#endif
