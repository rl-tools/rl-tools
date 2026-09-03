#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "../../../devices/dummy.h"
#include "on_policy_runner.h"
#include "../../environments/batch/operations_cuda.h"
#include "../episodes/operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
    void record_step(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated);
    template <typename DEV_SPEC, typename DATASET_SPEC, typename RESET_SPEC>
    void record_reset(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset);
    namespace rl::components::on_policy_runner{
        template <typename DEV_SPEC, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        void sample_actions(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
        __global__
        void record_step_kernel(DEVICE device, Dataset<DATASET_SPEC> dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC> rewards, const Tensor<TERMINATED_SPEC> terminated, const Tensor<TRUNCATED_SPEC> truncated){
            using TI = typename DATASET_SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS){
                record_step_env(device, dataset, step_i, rewards, terminated, truncated, env_i);
            }
        }
        template <typename DEVICE, typename DATASET_SPEC, typename RESET_SPEC>
        __global__
        void record_reset_kernel(DEVICE device, Dataset<DATASET_SPEC> dataset, const Tensor<RESET_SPEC> reset){
            using TI = typename DATASET_SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS){
                record_reset_env(device, dataset, reset, env_i);
            }
        }
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename REWARD_SPEC, typename TERMINATED_SPEC, typename TRUNCATED_SPEC>
    void record_step(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, typename DATASET_SPEC::TI step_i, const Tensor<REWARD_SPEC>& rewards, const Tensor<TERMINATED_SPEC>& terminated, const Tensor<TRUNCATED_SPEC>& truncated){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, REWARD_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TERMINATED_SPEC>());
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, TRUNCATED_SPEC>());
        utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::record_step: step index outside the dataset");
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(DATASET_SPEC::SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::on_policy_runner::record_step_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, step_i, rewards, terminated, truncated);
        check_status(device);
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename RESET_SPEC>
    void record_reset(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Tensor<RESET_SPEC>& reset){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::components::on_policy_runner::check_batched_flags<DATASET_SPEC, RESET_SPEC>());
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(DATASET_SPEC::SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::on_policy_runner::record_reset_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, reset);
        check_status(device);
    }
    namespace rl::components::on_policy_runner{
        template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        __global__
        void sample_actions_kernel(DEVICE device, Dataset<DATASET_SPEC> dataset, const Matrix<LOG_STD_SPEC> log_std, Tensor<STEP_ACTIONS_SPEC> step_actions, typename DATASET_SPEC::TI step_i, RNG rng){
            using TI = typename DATASET_SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS){
                auto& rng_state = get(rng.states, 0, env_i);
                sample_actions_env(device, dataset, log_std, step_actions, step_i, env_i, rng_state);
            }
        }
        template <typename DEV_SPEC, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        void sample_actions(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(DATASET_SPEC::SPEC::N_ENVIRONMENTS, BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            sample_actions_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, log_std, step_actions, step_i, rng);
            check_status(device);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
