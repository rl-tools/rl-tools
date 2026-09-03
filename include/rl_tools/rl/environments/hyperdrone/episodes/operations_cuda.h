#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_OPERATIONS_CUDA_H

#include "operations_cpu.h"
#include "../operations_cuda.h"

#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::episodes::cuda {
        template <typename DEVICE, typename SPEC>
        __global__ void synchronized_due_kernel(DEVICE device, Episodes<SPEC> episodes){
            using TI = typename DEVICE::index_t;
            if(threadIdx.x == 0 && blockIdx.x == 0){
                bool any_due = false;
                for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                    any_due = any_due || _due(device, episodes, instance_i);
                }
                set(device, episodes.synchronized_due, any_due, 0);
            }
        }
        template <typename DEVICE, typename SPEC, bool WITH_LOG, typename LOG>
        __global__ void begin_step_kernel(DEVICE device, Episodes<SPEC> episodes, LOG log, typename SPEC::TI step_i){
            using T = typename SPEC::T;
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                const bool due = SPEC::SYNCHRONIZED ? get(device, episodes.synchronized_due, 0) : _due(device, episodes, instance_i);
                T finished_length, finished_return;
                TI finished_reason;
                _begin_step(device, episodes, instance_i, due, finished_length, finished_return, finished_reason);
                if constexpr(WITH_LOG){
                    set(device, log.finished_length, finished_length, step_i, instance_i);
                    set(device, log.finished_return, finished_return, step_i, instance_i);
                    set(device, log.finished_reason, finished_reason, step_i, instance_i);
                }
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
        template <typename DEV_SPEC, typename SPEC, bool WITH_LOG, typename LOG>
        void _apply_resets(devices::CUDA<DEV_SPEC>& device, Episodes<SPEC>& episodes, LOG& log, typename SPEC::TI step_i){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            if constexpr(SPEC::SYNCHRONIZED){
                synchronized_due_kernel<decltype(tag_device), SPEC><<<dim3(1), dim3(1), 0, device.stream>>>(tag_device, episodes);
                check_status(device);
            }
            begin_step_kernel<decltype(tag_device), SPEC, WITH_LOG><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, log, step_i);
            check_status(device);
        }
    }

    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
    void begin_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        rl::environments::hyperdrone::episodes::Log<SPEC, 1> unused_log{};
        rl::environments::hyperdrone::episodes::cuda::_apply_resets<DEV_SPEC, SPEC, false>(device, episodes, unused_log, (TI)0);
        sample_initial_parameters(device, environment, parameters, episodes.reset, rng);
        sample_initial_state(device, environment, parameters, states, episodes.reset, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename SPEC::TI STEPS, typename RNG>
    void begin_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log, typename SPEC::TI step_i, RNG& rng){
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        utils::assert_exit(device, step_i < STEPS, "hyperdrone::episodes::begin_step: step index outside the log");
        rl::environments::hyperdrone::episodes::cuda::_apply_resets<DEV_SPEC, SPEC, true>(device, episodes, log, step_i);
        sample_initial_parameters(device, environment, parameters, episodes.reset, rng);
        sample_initial_state(device, environment, parameters, states, episodes.reset, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::end_step_kernel<decltype(tag_device), SPEC, true><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, rewards);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::end_step_kernel<decltype(tag_device), SPEC, false><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, episodes.episode_return);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::force_reset_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, const Tensor<MASK_SPEC>& mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::INSTANCES);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::force_reset_mask_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, mask);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
