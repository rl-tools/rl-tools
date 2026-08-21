#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_CUDA_BATCH_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_CUDA_BATCH_H

#include "operations_generic_batch.h"
#include "../../devices/cuda.h"

#include <type_traits>

// kernel-mapped defaults of the batch environment verbs: one thread per instance, the
// environment passed by value (device-visible for POD environments), per-instance RNG states

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::batch::cuda{
        template <typename TI>
        static constexpr TI BLOCKSIZE = 32;

        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename PARAMETER_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                sample_initial_parameters(device, env, get_ref(device, parameters, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_state_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<RESET_SPEC> reset_mask, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES && get(device, reset_mask, instance_i)){
                auto& rng_state = get(rng.states, 0, instance_i);
                sample_initial_state(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
        __global__ void observe_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC> observations, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                auto observation_slice = view(device, observations, instance_i);
                auto observation_matrix = matrix_view(device, observation_slice);
                observe(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), observation_type, observation_matrix, rng_state);
            }
        }
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
        __global__ void step_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action_matrix;
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
                }
                step(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng_state);
            }
        }
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
        __global__ void reward_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, Tensor<REWARD_SPEC> rewards, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action_matrix;
                for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                    set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
                }
                set(device, rewards, reward(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng_state), instance_i);
            }
        }
        template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
        __global__ void terminated_kernel(DEVICE device, ENVIRONMENT env, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<TERMINATED_SPEC> terminated_flags, RNG rng){
            using TI = typename DEVICE::index_t;
            constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
            static_assert(RNG::NUM_RNGS >= INSTANCES, "Please increase the number of CUDA RNGs");
            TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < INSTANCES){
                auto& rng_state = get(rng.states, 0, instance_i);
                set(device, terminated_flags, terminated(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng_state), instance_i);
            }
        }
    }

    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(utils::typing::is_same_v<typename PARAMETER_SPEC::T, typename ENVIRONMENT::Parameters>);
        constexpr TI INSTANCES = get<0>(typename PARAMETER_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::sample_initial_parameters_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, reset_mask, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::sample_initial_state_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, states, reset_mask, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::observe_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, states, observation_type, observations, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::step_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, states, actions, next_states, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::reward_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, states, actions, next_states, rewards, rng);
        check_status(device);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename utils::typing::enable_if<std::is_trivially_copyable<ENVIRONMENT>::value, bool>::type = true, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(devices::CUDA<DEV_SPEC>& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        constexpr TI BLOCKSIZE = rl::environments::batch::cuda::BLOCKSIZE<TI>;
        constexpr TI N_BLOCKS = RL_TOOLS_DEVICES_CUDA_CEIL(INSTANCES, BLOCKSIZE);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::terminated_kernel<<<dim3(N_BLOCKS), dim3(BLOCKSIZE), 0, device.stream>>>(tag_device, env, parameters, states, terminated_flags, rng);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
