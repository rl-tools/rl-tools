#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "operations_cpu.h"
#include "../operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::hyperdrone::episodes::cuda {
        template <typename DEVICE, typename SPEC, typename DATASET_SPEC>
        __global__ void record_rollout_start_kernel(DEVICE device, const Episodes<SPEC> episodes, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset){
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                _record_rollout_start(device, episodes, dataset, instance_i);
            }
        }
        template <typename DEVICE, typename SPEC, typename REWARD_SPEC, typename DATASET_SPEC>
        __global__ void record_kernel(DEVICE device, const Episodes<SPEC> episodes, const Tensor<REWARD_SPEC> rewards, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, typename SPEC::TI step_i){
            using TI = typename DEVICE::index_t;
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                _record(device, episodes, rewards, dataset, step_i, instance_i);
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename DATASET_SPEC>
    void record_rollout_start(devices::CUDA<DEV_SPEC>& device, const rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::hyperdrone::episodes::check_dataset<SPEC, DATASET_SPEC>());
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::record_rollout_start_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, dataset);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename REWARD_SPEC, typename DATASET_SPEC>
    void record(devices::CUDA<DEV_SPEC>& device, const rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, const Tensor<REWARD_SPEC>& rewards, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename SPEC::TI step_i){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::hyperdrone::episodes::check_dataset<SPEC, DATASET_SPEC>());
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::hyperdrone::episodes::cuda::record_kernel<decltype(tag_device), SPEC><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, episodes, rewards, dataset, step_i);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
