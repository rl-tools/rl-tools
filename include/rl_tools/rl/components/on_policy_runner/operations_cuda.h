#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "../../../devices/dummy.h"
#include "on_policy_runner.h"
#include "../../environments/batch/operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        namespace detail{
            template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
            void prologue(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner);
            template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
            void epilogue(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, const Buffer<SPEC>& buffer, typename SPEC::TI step_i);
            template <typename DEV_SPEC, typename SPEC>
            void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner);
            template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
            void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask);
        }
        template <typename DEV_SPEC, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
        void sample_actions(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        namespace detail{
            template <typename T_SPEC>
            struct RunnerStateView {
                using SPEC = T_SPEC;
                using RUNNER = rl::components::OnPolicyRunner<SPEC>;
                using TI = typename SPEC::TI;
                using EPISODE_T = typename RUNNER::EPISODE_T;
                Tensor<typename RUNNER::COUNTER_SPEC> episode_step;
                Tensor<typename RUNNER::FLAG_SPEC> reset;
                Tensor<typename RUNNER::VALUE_SPEC> episode_return;
                Tensor<typename RUNNER::COUNTER_SPEC> completed_episode_length;
                Tensor<typename RUNNER::VALUE_SPEC> completed_episode_return;
                Tensor<typename RUNNER::REASON_SPEC> completed_episode_reason;
                typename SPEC::TI episode_step_limit;
            };
            template <typename SPEC>
            RunnerStateView<SPEC> state_view(rl::components::OnPolicyRunner<SPEC>& runner){
                return {runner.episode_step, runner.reset, runner.episode_return, runner.completed_episode_length, runner.completed_episode_return, runner.completed_episode_reason, runner.episode_step_limit};
            }
            template <typename DEVICE, typename DATASET_SPEC, typename SPEC>
            __global__ void prologue_kernel(DEVICE device, Dataset<DATASET_SPEC> dataset, RunnerStateView<SPEC> runner){
                using TI = typename SPEC::TI;
                TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
                if(env_i < SPEC::N_ENVIRONMENTS){
                    prologue_env(device, dataset, runner, env_i);
                }
            }
            template <typename DEVICE, typename DATASET_SPEC, typename SPEC>
            __global__ void epilogue_kernel(DEVICE device, Dataset<DATASET_SPEC> dataset, RunnerStateView<SPEC> runner, const Buffer<SPEC> buffer, typename SPEC::TI step_i){
                using TI = typename SPEC::TI;
                TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
                if(env_i < SPEC::N_ENVIRONMENTS){
                    epilogue_env(device, dataset, runner, buffer, step_i, env_i);
                }
            }
            template <typename DEVICE, typename SPEC>
            __global__ void reset_kernel(DEVICE device, RunnerStateView<SPEC> runner){
                using TI = typename SPEC::TI;
                TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
                if(env_i < SPEC::N_ENVIRONMENTS){
                    reset_env(device, runner, env_i);
                }
            }
            template <typename DEVICE, typename SPEC, typename MASK_SPEC>
            __global__ void reset_kernel(DEVICE device, RunnerStateView<SPEC> runner, const Tensor<MASK_SPEC> mask){
                using TI = typename SPEC::TI;
                TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
                if(env_i < SPEC::N_ENVIRONMENTS && get(device, mask, env_i)){
                    reset_env(device, runner, env_i);
                }
            }
            template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
            void prologue(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner){
                using DEVICE = devices::CUDA<DEV_SPEC>;
                using TI = typename SPEC::TI;
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
                devices::cuda::TAG<DEVICE, true> tag_device{};
                prologue_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, state_view(runner));
                check_status(device);
            }
            template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
            void epilogue(devices::CUDA<DEV_SPEC>& device, Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, const Buffer<SPEC>& buffer, typename SPEC::TI step_i){
                using DEVICE = devices::CUDA<DEV_SPEC>;
                using TI = typename SPEC::TI;
                rl_tools::utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::epilogue: step index outside the dataset");
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
                devices::cuda::TAG<DEVICE, true> tag_device{};
                epilogue_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, state_view(runner), buffer, step_i);
                check_status(device);
            }
            template <typename DEV_SPEC, typename SPEC>
            void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner){
                using DEVICE = devices::CUDA<DEV_SPEC>;
                using TI = typename SPEC::TI;
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
                devices::cuda::TAG<DEVICE, true> tag_device{};
                reset_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state_view(runner));
                check_status(device);
            }
            template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
            void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask){
                using DEVICE = devices::CUDA<DEV_SPEC>;
                using TI = typename SPEC::TI;
                static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
                constexpr TI BLOCKSIZE = 32;
                constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
                devices::cuda::TAG<DEVICE, true> tag_device{};
                reset_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state_view(runner), mask);
                check_status(device);
            }
        }
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
