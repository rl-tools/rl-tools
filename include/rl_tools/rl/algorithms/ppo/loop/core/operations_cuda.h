#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_CUDA_H

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::loop::core::cuda{
        template <typename DEVICE, typename CONFIG, typename ENV_SPEC>
        __global__
        void init_environments_kernel(DEVICE device, Tensor<ENV_SPEC> envs){
            using TI = typename DEVICE::index_t;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS){
                auto& env = get_ref(device, envs, env_i);
                init(device, env);
            }
        }
        template <typename DEVICE, typename CONFIG, typename ENV_SPEC>
        __global__
        void malloc_environments_kernel(DEVICE device, Tensor<ENV_SPEC> envs){
            using TI = typename DEVICE::index_t;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS){
                auto& env = get_ref(device, envs, env_i);
                malloc(device, env);
            }
        }
    }
    template <typename DEV_SPEC, typename T_CONFIG>
    void malloc(devices::CUDA<DEV_SPEC>& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
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
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(T_CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::loop::core::cuda::malloc_environments_kernel<decltype(tag_device), T_CONFIG><<<grid, block, 0, device.stream>>>(tag_device, ts.envs);
        check_status(device);
    }
    template <typename DEV_SPEC, typename T_CONFIG>
    void free(devices::CUDA<DEV_SPEC>& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
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
        // L2F env free is a no-op, no kernel needed
    }
    template <typename DEV_SPEC, typename T_CONFIG>
    void init(devices::CUDA<DEV_SPEC>& device, rl::algorithms::ppo::loop::core::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using CONFIG = T_CONFIG;
        using TI = typename DEVICE::index_t;

        init(device, ts.rng, seed);

        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(CONFIG::CORE_PARAMETERS::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::loop::core::cuda::init_environments_kernel<decltype(tag_device), CONFIG><<<grid, block, 0, device.stream>>>(tag_device, ts.envs);
        check_status(device);

        init(device, ts.ppo, ts.actor_optimizer, ts.critic_optimizer, ts.rng);
        init(device, ts.on_policy_runner, ts.envs, ts.env_parameters, ts.ppo.actor, ts.rng);

        ts.step = 0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
