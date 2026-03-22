#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_LOOP_CORE_OPERATIONS_CUDA_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::ppo::loop::core::cuda{
        template <typename DEVICE, typename ENV_SPEC>
        __global__
        void init_environments_kernel(DEVICE device, Tensor<ENV_SPEC> envs){
            using TI = typename DEVICE::index_t;
            constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < N){
                auto& env = get_ref(device, envs, env_i);
                init(device, env);
            }
        }
        template <typename DEVICE, typename ENV_SPEC>
        __global__
        void malloc_environments_kernel(DEVICE device, Tensor<ENV_SPEC> envs){
            using TI = typename DEVICE::index_t;
            constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < N){
                auto& env = get_ref(device, envs, env_i);
                malloc(device, env);
            }
        }
    }
    // CUDA overrides for the device-dependent helpers
    template <typename DEV_SPEC, typename ENV_SPEC>
    void malloc_environments(devices::CUDA<DEV_SPEC>& device, Tensor<ENV_SPEC>& envs){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(N, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::loop::core::cuda::malloc_environments_kernel<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, envs);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENV_SPEC>
    void free_environments(devices::CUDA<DEV_SPEC>& device, Tensor<ENV_SPEC>& envs){
        // L2F env free is a no-op, no kernel needed
    }
    template <typename DEV_SPEC, typename ENV_SPEC>
    void init_environments(devices::CUDA<DEV_SPEC>& device, Tensor<ENV_SPEC>& envs){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI N = ENV_SPEC::SHAPE::template GET<0>;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(N, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::algorithms::ppo::loop::core::cuda::init_environments_kernel<<<N_BLOCKS, BLOCKSIZE, 0, device.stream>>>(tag_device, envs);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "operations_generic.h"
#endif
