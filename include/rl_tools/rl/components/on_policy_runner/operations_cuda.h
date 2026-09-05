#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CUDA_H

#include "../../../devices/dummy.h"
#include "on_policy_runner.h"
#include "../../environments/batch/operations_cuda.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
    void record_transition(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, const rl::components::on_policy_runner::Buffer<SPEC>& buffer, typename SPEC::TI step_i);
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void reset_mask(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask);
    template <typename DEV_SPEC, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    void sample_actions(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng);
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void prologue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng);
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void epilogue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng, typename SPEC::TI step_i);
    template <typename DEV_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng);
    template <typename DEV_SPEC, typename SPEC, typename BATCH_SPEC, typename MASK_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, const Tensor<MASK_SPEC>& mask, RNG& rng);
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        template <typename T_SPEC>
        struct RunnerStateView {
            using SPEC = T_SPEC;
            using RUNNER = rl::components::OnPolicyRunner<SPEC>;
            using TI = typename SPEC::TI;
            Tensor<typename RUNNER::COUNTER_SPEC> episode_step;
            Tensor<typename RUNNER::FLAG_SPEC> reset;
            decltype(RUNNER::env_parameters) env_parameters;
            decltype(RUNNER::states) states;
        };
    }
    template <typename SPEC>
    rl::components::on_policy_runner::RunnerStateView<SPEC> runner_state_view(rl::components::OnPolicyRunner<SPEC>& runner){
        return {runner.episode_step, runner.reset, runner.env_parameters, runner.states};
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC>
    __global__ void epilogue_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, const rl::components::on_policy_runner::Buffer<SPEC> buffer, typename SPEC::TI step_i){
        using TI = typename SPEC::TI;
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS){
            record_transition(device, dataset, runner, get(device, buffer.rewards, env_i), get(device, buffer.terminated, env_i), step_i, env_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC>
    __global__ void reset_kernel(DEVICE device, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, const Tensor<MASK_SPEC> mask){
        using TI = typename SPEC::TI;
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS && get(device, mask, env_i)){
            set(device, runner.reset, true, env_i);
            set(device, runner.episode_step, (typename SPEC::TI)0, env_i);
        }
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC>
    void record_transition(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, const rl::components::on_policy_runner::Buffer<SPEC>& buffer, typename SPEC::TI step_i){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        rl_tools::utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::epilogue: step index outside the dataset");
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        epilogue_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, runner_state_view(runner), buffer, step_i);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename MASK_SPEC>
    void reset_mask(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, const Tensor<MASK_SPEC>& mask){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename SPEC::TI;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        reset_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, runner_state_view(runner), mask);
        check_status(device);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    __global__
    void sample_actions_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, const Matrix<LOG_STD_SPEC> log_std, Tensor<STEP_ACTIONS_SPEC> step_actions, typename DATASET_SPEC::TI step_i, RNG rng){
        using TI = typename DATASET_SPEC::TI;
        TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < DATASET_SPEC::SPEC::N_ENVIRONMENTS){
            auto& rng_state = get(rng.states, 0, env_i);
            sample_actions_env(device, dataset, log_std, step_actions, step_i, env_i, rng_state);
        }
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename LOG_STD_SPEC, typename STEP_ACTIONS_SPEC, typename RNG>
    void sample_actions(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const Matrix<LOG_STD_SPEC>& log_std, Tensor<STEP_ACTIONS_SPEC>& step_actions, typename DATASET_SPEC::TI step_i, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(RNG::NUM_RNGS >= DATASET_SPEC::SPEC::N_ENVIRONMENTS, "the runner needs one RNG state per environment instance");
        constexpr TI BLOCKSIZE = 32;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(DATASET_SPEC::SPEC::N_ENVIRONMENTS, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        sample_actions_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, dataset, log_std, step_actions, step_i, rng);
        check_status(device);
    }

    template <typename DEVICE, typename DATASET_SPEC, typename ENVIRONMENT, typename PARAMETERS, typename STATE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void observe_instance(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, ENVIRONMENT& environment, PARAMETERS& parameters, STATE& state, typename DATASET_SPEC::TI row_i, typename DATASET_SPEC::TI env_i, RNG& rng){
        using SPEC = typename DATASET_SPEC::SPEC;
        const auto pos = row_i * SPEC::N_ENVIRONMENTS + env_i;
        auto observation = view(device, dataset.all_observations, pos);
        auto observation_matrix = matrix_view(device, observation);
        observe(device, environment, parameters, state, typename SPEC::OBSERVATION{}, observation_matrix, rng);
        auto observation_privileged = view(device, dataset.all_observations_privileged, pos);
        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            auto observation_privileged_matrix = matrix_view(device, observation_privileged);
            observe(device, environment, parameters, state, typename SPEC::OBSERVATION_PRIVILEGED{}, observation_privileged_matrix, rng);
        }
        else{
            copy(device, device, observation, observation_privileged);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENV_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset_instance(DEVICE& device, rl::components::on_policy_runner::RunnerStateView<SPEC>& runner, Tensor<ENV_SPEC>& environments, typename SPEC::TI env_i, RNG& rng){
        auto& environment = get_ref(device, environments, env_i);
        auto& parameters = get_ref(device, runner.env_parameters, env_i);
        auto& state = get_ref(device, runner.states, env_i);
        sample_initial_parameters(device, environment, parameters, rng);
        sample_initial_state(device, environment, parameters, state, rng);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENV_SPEC, typename RNG>
    __global__ void prologue_independent_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, Tensor<ENV_SPEC> environments, RNG rng){
        const typename SPEC::TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS){
            set(dataset.reset, env_i, 0, get(device, runner.reset, env_i));
            auto& rng_state = get(rng.states, 0, env_i);
            observe_instance(device, dataset, get_ref(device, environments, env_i), get_ref(device, runner.env_parameters, env_i), get_ref(device, runner.states, env_i), 0, env_i, rng_state);
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENV_SPEC, typename RNG>
    __global__ void epilogue_independent_kernel(DEVICE device, rl::components::on_policy_runner::Dataset<DATASET_SPEC> dataset, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, rl::components::on_policy_runner::Buffer<SPEC> buffer, Tensor<ENV_SPEC> environments, RNG rng, typename SPEC::TI step_i){
        const typename SPEC::TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS){
            auto& environment = get_ref(device, environments, env_i);
            auto& parameters = get_ref(device, runner.env_parameters, env_i);
            auto& state = get_ref(device, runner.states, env_i);
            auto& rng_state = get(rng.states, 0, env_i);
            auto action = matrix_view(device, view(device, buffer.actions, env_i));
            typename SPEC::BATCH_ENVIRONMENT::State next_state;
            step(device, environment, parameters, state, action, next_state, rng_state);
            const typename SPEC::T transition_reward = reward(device, environment, parameters, state, action, next_state, rng_state);
            const bool transition_terminated = terminated(device, environment, parameters, next_state, rng_state);
            get_ref(device, buffer.next_states, env_i) = next_state;
            set(device, buffer.rewards, transition_reward, env_i);
            set(device, buffer.terminated, transition_terminated, env_i);
            state = next_state;
            record_transition(device, dataset, runner, transition_reward, transition_terminated, step_i, env_i);
            if(get(device, runner.reset, env_i)){
                reset_instance(device, runner, environments, env_i, rng_state);
            }
            observe_instance(device, dataset, environment, parameters, state, step_i + 1, env_i, rng_state);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENV_SPEC, typename RNG>
    __global__ void reset_independent_kernel(DEVICE device, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, Tensor<ENV_SPEC> environments, RNG rng){
        const typename SPEC::TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS){
            set(device, runner.reset, true, env_i);
            set(device, runner.episode_step, (typename SPEC::TI)0, env_i);
            auto& rng_state = get(rng.states, 0, env_i);
            reset_instance(device, runner, environments, env_i, rng_state);
        }
    }
    template <typename DEVICE, typename SPEC, typename ENV_SPEC, typename MASK_SPEC, typename RNG>
    __global__ void reset_independent_kernel(DEVICE device, rl::components::on_policy_runner::RunnerStateView<SPEC> runner, Tensor<ENV_SPEC> environments, const Tensor<MASK_SPEC> mask, RNG rng){
        const typename SPEC::TI env_i = threadIdx.x + blockIdx.x * blockDim.x;
        if(env_i < SPEC::N_ENVIRONMENTS && get(device, mask, env_i)){
            set(device, runner.reset, true, env_i);
            set(device, runner.episode_step, (typename SPEC::TI)0, env_i);
            auto& rng_state = get(rng.states, 0, env_i);
            reset_instance(device, runner, environments, env_i, rng_state);
        }
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void prologue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng){
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, rl::environments::batch::Independent<BATCH_SPEC>>);
        static_assert(utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>);
        static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "the runner needs one RNG state per environment instance");
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        prologue_independent_kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, 32), 32, 0, device.stream>>>(tag_device, dataset, runner_state_view(runner), environment.environments, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename DATASET_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void epilogue(devices::CUDA<DEV_SPEC>& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng, typename SPEC::TI step_i){
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, rl::environments::batch::Independent<BATCH_SPEC>>);
        static_assert(utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>);
        static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "the runner needs one RNG state per environment instance");
        utils::assert_exit(device, step_i < DATASET_SPEC::STEPS_PER_ENV, "on_policy_runner::epilogue: step index outside the dataset");
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        epilogue_independent_kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, 32), 32, 0, device.stream>>>(tag_device, dataset, runner_state_view(runner), buffer, environment.environments, rng, step_i);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename BATCH_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, RNG& rng){
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, rl::environments::batch::Independent<BATCH_SPEC>>);
        static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "the runner needs one RNG state per environment instance");
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        reset_independent_kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, 32), 32, 0, device.stream>>>(tag_device, runner_state_view(runner), environment.environments, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename SPEC, typename BATCH_SPEC, typename MASK_SPEC, typename RNG>
    void reset(devices::CUDA<DEV_SPEC>& device, rl::components::OnPolicyRunner<SPEC>& runner, rl::environments::batch::Independent<BATCH_SPEC>& environment, const Tensor<MASK_SPEC>& mask, RNG& rng){
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, rl::environments::batch::Independent<BATCH_SPEC>>);
        static_assert(RNG::NUM_RNGS >= SPEC::N_ENVIRONMENTS, "the runner needs one RNG state per environment instance");
        static_assert(length(typename MASK_SPEC::SHAPE{}) == 1 && get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::N_ENVIRONMENTS);
        devices::cuda::TAG<devices::CUDA<DEV_SPEC>, true> tag_device{};
        reset_independent_kernel<<<RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::N_ENVIRONMENTS, 32), 32, 0, device.stream>>>(tag_device, runner_state_view(runner), environment.environments, mask, rng);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
