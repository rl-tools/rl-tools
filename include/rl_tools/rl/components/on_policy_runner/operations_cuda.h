#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "../../../devices/dummy.h"
#include "on_policy_runner.h"
#include "../../environments/batch/operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void begin_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng);
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename REWARD_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, const Tensor<REWARD_SPEC>& rewards, RNG& rng);
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng);
    template <typename DEV_SPEC, typename SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner);
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask);
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
        namespace detail{
            template <typename T_SPEC>
            struct EpisodeStateView {
                using SPEC = T_SPEC;
                using RUNNER = rl::components::OnPolicyRunner<SPEC>;
                using TI = typename SPEC::TI;
                using EPISODE_T = typename RUNNER::EPISODE_T;
                Tensor<typename RUNNER::COUNTER_SPEC> episode_step;
                Tensor<typename RUNNER::FLAG_SPEC> terminated;
                Tensor<typename RUNNER::FLAG_SPEC> truncated;
                Tensor<typename RUNNER::FLAG_SPEC> reset;
                Tensor<typename RUNNER::FLAG_SPEC> forced;
                Tensor<typename RUNNER::VALUE_SPEC> episode_return;
                Tensor<typename RUNNER::REASON_SPEC> end_reason;
                Tensor<typename RUNNER::FLAG_SPEC> finished;
                Tensor<typename RUNNER::COUNTER_SPEC> finished_length;
                Tensor<typename RUNNER::VALUE_SPEC> finished_return;
                Tensor<typename RUNNER::REASON_SPEC> finished_reason;
                typename SPEC::TI episode_step_limit;
            };
            template <typename SPEC>
            EpisodeStateView<SPEC> episode_state_view(rl::components::OnPolicyRunner<SPEC>& runner){
                return {runner.episode_step, runner.terminated, runner.truncated, runner.reset, runner.forced, runner.episode_return, runner.end_reason, runner.finished, runner.finished_length, runner.finished_return, runner.finished_reason, runner.episode_step_limit};
            }
        }
        template <typename DEVICE, typename SPEC>
        __global__
        void begin_step_kernel(DEVICE device, detail::EpisodeStateView<SPEC> state){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                detail::begin_episode_step(device, state, env_i, detail::episode_due(device, state, env_i));
            }
        }
        template <typename DEVICE, typename SPEC>
        __global__
        void begin_step_synchronized_kernel(DEVICE device, detail::EpisodeStateView<SPEC> state){
            using TI = typename SPEC::TI;
            bool any_due = false;
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                any_due = any_due || detail::episode_due(device, state, env_i);
            }
            for(TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                detail::begin_episode_step(device, state, env_i, any_due);
            }
        }
        template <typename DEVICE, typename SPEC, bool WITH_REWARDS, typename REWARD_SPEC>
        __global__
        void end_step_kernel(DEVICE device, detail::EpisodeStateView<SPEC> state, const Tensor<REWARD_SPEC> rewards){
            using T = typename rl::components::OnPolicyRunner<SPEC>::EPISODE_T;
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                T reward = 0;
                if constexpr(WITH_REWARDS){
                    reward = get(device, rewards, env_i);
                }
                detail::end_episode_step(device, state, env_i, reward);
            }
        }
        template <typename DEVICE, typename SPEC>
        __global__
        void force_reset_kernel(DEVICE device, detail::EpisodeStateView<SPEC> state){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                detail::force_reset(device, state, env_i);
            }
        }
        template <typename DEVICE, typename SPEC, typename MASK_SPEC>
        __global__
        void force_reset_mask_kernel(DEVICE device, detail::EpisodeStateView<SPEC> state, const Tensor<MASK_SPEC> mask){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS && get(device, mask, env_i)){
                detail::force_reset(device, state, env_i);
            }
        }
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
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void begin_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto state = rl::components::on_policy_runner::detail::episode_state_view(runner);
        if constexpr(SPEC::SYNCHRONIZED){
            rl::components::on_policy_runner::begin_step_synchronized_kernel<<<1, 1, 0, device.stream>>>(tag_device, state);
        }
        else{
            rl::components::on_policy_runner::begin_step_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state);
        }
        check_status(device);
        sample_initial_parameters(device, environment, runner.env_parameters, runner.reset, rng);
        sample_initial_state(device, environment, runner.env_parameters, runner.states, runner.reset, rng);
    }
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename REWARD_SPEC, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, const Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        terminated(device, environment, runner.env_parameters, runner.states, runner.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto state = rl::components::on_policy_runner::detail::episode_state_view(runner);
        rl::components::on_policy_runner::end_step_kernel<decltype(tag_device), SPEC, true><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state, rewards);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename ENVIRONMENT, typename RNG>
    void end_step(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, ENVIRONMENT& environment, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        terminated(device, environment, runner.env_parameters, runner.states, runner.terminated, rng);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto state = rl::components::on_policy_runner::detail::episode_state_view(runner);
        rl::components::on_policy_runner::end_step_kernel<decltype(tag_device), SPEC, false><<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state, runner.episode_return);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto state = rl::components::on_policy_runner::detail::episode_state_view(runner);
        rl::components::on_policy_runner::force_reset_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void force_reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        auto state = rl::components::on_policy_runner::detail::episode_state_view(runner);
        rl::components::on_policy_runner::force_reset_mask_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, state, mask);
        check_status(device);
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
