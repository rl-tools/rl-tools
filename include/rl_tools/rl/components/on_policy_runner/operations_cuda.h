#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "../../../devices/dummy.h"
#include "operations_generic.h"
#include "on_policy_runner.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        template<typename DEVICE, typename SPEC, typename ENV_SPEC, typename PARAM_SPEC>
        __global__
        void init_kernel(DEVICE device, rl::components::OnPolicyRunner<SPEC> runner, Tensor<ENV_SPEC> environments, Tensor<PARAM_SPEC> parameters){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                auto& src_env = get_ref(device, environments, env_i);
                auto& src_params = get_ref(device, parameters, env_i);
                set(runner.environments, 0, env_i, src_env);
                set(runner.env_parameters, 0, env_i, src_params);
            }
        }
        template<typename DEVICE, typename OBS_PRIV_SPEC, typename OBS_SPEC, typename SPEC, typename RNG>
        __global__
        void prologue_kernel(DEVICE device, Tensor<OBS_PRIV_SPEC> observations_privileged, Tensor<OBS_SPEC> observations, rl::components::OnPolicyRunner<SPEC> runner, RNG rng){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "Please increase the number of CUDA RNGs");
            if(env_i < SPEC::N_ENVIRONMENTS){
                auto& rng_state = get(rng.states, 0, env_i);
                per_env::prologue(device, observations_privileged, observations, runner, rng_state, env_i);
            }
        }
        template<typename DEV_SPEC, typename OBS_PRIV_SPEC, typename OBS_SPEC, typename SPEC, typename RNG>
        void prologue(devices::CUDA<DEV_SPEC>& device, Tensor<OBS_PRIV_SPEC>& observations_privileged, Tensor<OBS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, RNG& rng, typename devices::CUDA<DEV_SPEC>::index_t step_i){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            prologue_kernel<<<grid, block, 0, device.stream>>>(tag_device, observations_privileged, observations, runner, rng);
            check_status(device);
        }
        // CUDA + ArrayENGINE: resolves ambiguity with generic ArrayENGINE overload
        template<typename DEV_SPEC, typename OBS_PRIV_SPEC, typename OBS_SPEC, typename SPEC, typename ARRAY_SPEC>
        void prologue(devices::CUDA<DEV_SPEC>& device, Tensor<OBS_PRIV_SPEC>& observations_privileged, Tensor<OBS_SPEC>& observations, rl::components::OnPolicyRunner<SPEC>& runner, devices::generic::random::ArrayENGINE<ARRAY_SPEC>& rng, typename devices::CUDA<DEV_SPEC>::index_t step_i){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            prologue_kernel<<<grid, block, 0, device.stream>>>(tag_device, observations_privileged, observations, runner, rng);
            check_status(device);
        }
        template<typename DEVICE, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG>
        __global__
        void epilogue_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC> runner, Matrix<ACTIONS_MEAN_SPEC> actions_mean, Matrix<ACTIONS_SPEC> actions, Matrix<ACTION_LOG_STD_SPEC> action_log_std, RNG rng, typename DATASET_SPEC::SPEC::TI step_i){
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "Please increase the number of CUDA RNGs");
            if(env_i < SPEC::N_ENVIRONMENTS){
                auto& rng_state = get(rng.states, 0, env_i);
                TI pos = step_i * SPEC::N_ENVIRONMENTS + env_i;
                per_env::epilogue(device, dataset, runner, actions_mean, actions, action_log_std, rng_state, pos, env_i);
            }
        }
        template<typename DEV_SPEC, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename RNG>
        void epilogue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_MEAN_SPEC>& actions_mean, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, RNG& rng, typename devices::CUDA<DEV_SPEC>::index_t step_i){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            epilogue_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, runner, actions_mean, actions, action_log_std, rng, step_i);
            check_status(device);
        }
        // CUDA + ArrayENGINE: resolves ambiguity with generic ArrayENGINE overload
        template<typename DEV_SPEC, typename DATASET_SPEC, typename ACTIONS_MEAN_SPEC, typename ACTIONS_SPEC, typename ACTION_LOG_STD_SPEC, typename ARRAY_SPEC>
        void epilogue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, Matrix<ACTIONS_MEAN_SPEC>& actions_mean, Matrix<ACTIONS_SPEC>& actions, Matrix<ACTION_LOG_STD_SPEC>& action_log_std, devices::generic::random::ArrayENGINE<ARRAY_SPEC>& rng, typename devices::CUDA<DEV_SPEC>::index_t step_i){
            using DEVICE = devices::CUDA<DEV_SPEC>;
            using SPEC = typename DATASET_SPEC::SPEC;
            using TI = typename DEVICE::index_t;
            constexpr TI BLOCKSIZE = 32;
            constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
            dim3 grid(N_BLOCKS);
            dim3 block(BLOCKSIZE);
            devices::cuda::TAG<DEVICE, true> tag_device{};
            epilogue_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, runner, actions_mean, actions, action_log_std, rng, step_i);
            check_status(device);
        }
        template<typename DEVICE, typename DATASET_SPEC, typename SPEC, typename RNG>
        __global__
        void final_observation_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::OnPolicyRunner<SPEC> runner, RNG rng){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                auto& rng_state = get(rng.states, 0, env_i);
                auto& env = get(runner.environments, 0, env_i);
                auto& state = get(runner.states, 0, env_i);
                auto& parameters = get(runner.env_parameters, 0, env_i);
                auto obs_slice = view(device, dataset.all_observations, (TI)(DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i));
                auto obs_matrix = matrix_view(device, obs_slice);
                observe(device, env, parameters, state, typename SPEC::ENVIRONMENT::Observation{}, obs_matrix, rng_state);
                auto obs_priv_slice = view(device, dataset.all_observations_privileged, (TI)(DATASET_SPEC::STEPS_PER_ENV * SPEC::N_ENVIRONMENTS + env_i));
                auto obs_priv_matrix = matrix_view(device, obs_priv_slice);
                observe(device, env, parameters, state, typename SPEC::ENVIRONMENT::ObservationPrivileged{}, obs_priv_matrix, rng_state);
            }
        }
        template<typename DEVICE, typename DATASET_SPEC, typename SPEC>
        __global__
        void copy_truncated_to_reset_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::OnPolicyRunner<SPEC> runner){
            using TI = typename SPEC::TI;
            TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(env_i < SPEC::N_ENVIRONMENTS){
                set(dataset.reset, env_i, 0, get(runner.truncated, 0, env_i));
            }
        }
    }
    template <typename DEV_SPEC, typename SPEC, typename ENV_SPEC, typename PARAM_SPEC, typename ACTOR, typename RNG>
    void init(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, Tensor<ENV_SPEC> environments, Tensor<PARAM_SPEC> parameters, ACTOR& actor, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        set_all(device, runner.episode_step, 0);
        set_all(device, runner.episode_return, 0);
        set_all(device, runner.truncated, true);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::components::on_policy_runner::init_kernel<<<grid, block, 0, device.stream>>>(tag_device, runner, environments, parameters);
        check_status(device);
        reset(device, actor, runner.policy_state, rng);
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename ACTOR, typename ACTOR_BUFFER, typename RNG>
    void collect(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<typename DATASET_SPEC::SPEC>& runner, ACTOR& actor, ACTOR_BUFFER& policy_eval_buffers, RNG& rng){
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using SPEC = typename DATASET_SPEC::SPEC;
        using T = typename SPEC::TYPE_POLICY::DEFAULT;
        using TI = typename SPEC::TI;
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        dim3 grid(N_BLOCKS);
        dim3 block(BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            set_all(device, runner.truncated, true);
        }
        rl::components::on_policy_runner::copy_truncated_to_reset_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, runner);
        check_status(device);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            auto actions_mean            = view(device, dataset.actions_mean               , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto actions                 = view(device, dataset.actions                    , matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::ENVIRONMENT::ACTION_DIM>()                , step_i*SPEC::N_ENVIRONMENTS, 0);
            auto observations_privileged = view_range(device, dataset.all_observations_privileged, step_i*SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto observations            = view_range(device, dataset.all_observations          , step_i*SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto truncated_view = view(device, runner.truncated);
            Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(truncated_view)>>> mode_reset_mask;
            mode_reset_mask.mask = truncated_view;
            reset(device, actor, runner.policy_state, rng, mode_reset_mask);
            rl::components::on_policy_runner::prologue(device, observations_privileged, observations, runner, rng, step_i);
            using OBS_SHAPE = typename SPEC::ENVIRONMENT::Observation::SHAPE;
            using EVAL_INPUT_SHAPE = tensor::Prepend<OBS_SHAPE, SPEC::N_ENVIRONMENTS>;
            auto observations_reshaped = reshape_row_major(device, observations, EVAL_INPUT_SHAPE{});
            auto actions_mean_tensor = to_tensor(device, actions_mean);
            Mode<mode::Rollout<>> mode;
            evaluate_step(device, actor, observations_reshaped, runner.policy_state, actions_mean_tensor, policy_eval_buffers, rng, mode);
            auto& last_layer = get_last_layer(actor);
            auto log_std = matrix_view(device, last_layer.log_std.parameters);
            rl::components::on_policy_runner::epilogue(device, dataset, runner, actions_mean, actions, log_std, rng, step_i);
        }
        rl::components::on_policy_runner::final_observation_kernel<<<grid, block, 0, device.stream>>>(tag_device, dataset, runner, rng);
        check_status(device);
        runner.step += SPEC::N_ENVIRONMENTS * DATASET_SPEC::STEPS_PER_ENV;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#include "operations_generic.h"

#endif
