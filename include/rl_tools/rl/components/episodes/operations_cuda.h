#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_EPISODES_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_EPISODES_OPERATIONS_CUDA_H

#include "operations_cpu.h"

#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::components::episodes::cuda {
        template <typename DEVICE, typename SPEC>
        __global__ void begin_step_kernel(DEVICE device, Episodes<SPEC> episodes){
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                _begin_step(device, episodes, instance_i, _due(device, episodes, instance_i));
            }
        }
        template <typename DEVICE, typename SPEC>
        __global__ void begin_step_synchronized_kernel(DEVICE device, Episodes<SPEC> episodes){
            using TI = typename DEVICE::index_t;
            bool any_due = false;
            for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                any_due = any_due || _due(device, episodes, instance_i);
            }
            for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                _begin_step(device, episodes, instance_i, any_due);
            }
        }
        template <typename DEVICE, typename SPEC, bool WITH_REWARDS, typename REWARD_SPEC>
        __global__ void end_step_kernel(DEVICE device, Episodes<SPEC> episodes, const Tensor<REWARD_SPEC> rewards){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                T reward = 0;
                if constexpr(WITH_REWARDS){
                    reward = get(device, rewards, instance_i);
                }
                _end_step(device, episodes, instance_i, reward);
            }
        }
        template <typename DEVICE, typename SPEC>
        __global__ void force_reset_kernel(DEVICE device, Episodes<SPEC> episodes){
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                _force(device, episodes, instance_i);
            }
        }
        template <typename DEVICE, typename SPEC, typename MASK_SPEC>
        __global__ void force_reset_mask_kernel(DEVICE device, Episodes<SPEC> episodes, const Tensor<MASK_SPEC> mask){
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES && get(device, mask, instance_i)){
                _force(device, episodes, instance_i);
            }
        }
    }

    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
    void begin_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::components::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::components::episodes::check_environment<ENVIRONMENT, SPEC>());
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        if constexpr(SPEC::SYNCHRONIZED){
            rl::components::episodes::cuda::begin_step_synchronized_kernel<decltype(tag_device), SPEC><<<1, 1, 0, device.stream>>>(tag_device, episodes);
        }
        else{
            rl::components::episodes::cuda::begin_step_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes);
        }
        check_status(device);
        sample_initial_parameters(device, environment, parameters, episodes.reset, rng);
        sample_initial_state(device, environment, parameters, states, episodes.reset, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::components::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::components::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::episodes::cuda::end_step_kernel<decltype(tag_device), SPEC, true><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, rewards);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::components::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::components::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::episodes::cuda::end_step_kernel<decltype(tag_device), SPEC, false><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, episodes.episode_return);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::episodes::Episodes<SPEC>& episodes){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::episodes::cuda::force_reset_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::episodes::Episodes<SPEC>& episodes, const Tensor<MASK_SPEC>& mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::INSTANCES);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::episodes::cuda::force_reset_mask_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, mask);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
