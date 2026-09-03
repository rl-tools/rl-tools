#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_CUDA_H

#include "operations_generic.h"

#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace rl::environments::batch::cuda {
        template <typename DEVICE, typename SPEC>
        __global__ void malloc_kernel(DEVICE device, Independent<SPEC> batch){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                malloc(device, get_ref(device, batch.environments, instance_i));
            }
        }

        template <typename DEVICE, typename SPEC>
        __global__ void free_kernel(DEVICE device, Independent<SPEC> batch){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                free(device, get_ref(device, batch.environments, instance_i));
            }
        }

        template <typename DEVICE, typename SPEC>
        __global__ void init_kernel(DEVICE device, Independent<SPEC> batch){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                init(device, get_ref(device, batch.environments, instance_i));
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_parameters_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<RESET_SPEC> reset, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES && get(device, reset, instance_i)){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& rng_state = get(rng.states, 0, instance_i);
                sample_initial_parameters(device, environment, parameter, rng_state);
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
        __global__ void sample_initial_state_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<RESET_SPEC> reset, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES && get(device, reset, instance_i)){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto& rng_state = get(rng.states, 0, instance_i);
                sample_initial_state(device, environment, parameter, state, rng_state);
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
        __global__ void step_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto& next_state = get_ref(device, next_states, instance_i);
                auto action = matrix_view(device, view(device, actions, instance_i));
                auto& rng_state = get(rng.states, 0, instance_i);
                step(device, environment, parameter, state, action, next_state, rng_state);
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
        __global__ void reward_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<ACTION_SPEC> actions, Tensor<NEXT_STATE_SPEC> next_states, Tensor<REWARD_SPEC> rewards, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto& next_state = get_ref(device, next_states, instance_i);
                auto action = matrix_view(device, view(device, actions, instance_i));
                auto& rng_state = get(rng.states, 0, instance_i);
                set(device, rewards, reward(device, environment, parameter, state, action, next_state, rng_state), instance_i);
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
        __global__ void terminated_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, Tensor<TERMINATED_SPEC> terminated_flags, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto& rng_state = get(rng.states, 0, instance_i);
                set(device, terminated_flags, terminated(device, environment, parameter, state, rng_state), instance_i);
            }
        }

        template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION, typename OBSERVATION_SPEC, typename RNG>
        __global__ void observe_kernel(DEVICE device, Independent<SPEC> batch, Tensor<PARAMETER_SPEC> parameters, Tensor<STATE_SPEC> states, const OBSERVATION observation_type, Tensor<OBSERVATION_SPEC> observations, RNG rng){
            using TI = typename SPEC::TI;
            const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
            if(instance_i < SPEC::INSTANCES){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto observation_slice = view(device, observations, instance_i);
                auto observation = matrix_view(device, observation_slice);
                auto& rng_state = get(rng.states, 0, instance_i);
                observe(device, environment, parameter, state, observation_type, observation, rng_state);
            }
        }
    }

    template <typename SPEC>
    constexpr typename SPEC::TI batch_cuda_blocks(){
        return RL_TOOLS_DEVICES_CUDA_CEIL(SPEC::INSTANCES, 32);
    }

    template <typename DEV_SPEC, typename SPEC>
    void malloc(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        malloc(device, batch.environments);
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::malloc_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC>
    void free(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::free_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch);
        check_status(device);
        free(device, batch.environments);
    }

    template <typename DEV_SPEC, typename SPEC>
    void init(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::init_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::sample_initial_parameters_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, reset, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::sample_initial_state_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, states, reset, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::step_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, states, actions, next_states, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::reward_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, states, actions, next_states, rewards, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::terminated_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, states, terminated_flags, rng);
        check_status(device);
    }

    template <typename DEV_SPEC, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using DEVICE = devices::CUDA<DEV_SPEC>;
        static_assert(RNG::NUM_RNGS >= SPEC::INSTANCES, "the batch needs one RNG state per environment instance");
        devices::cuda::TAG<DEVICE, true> tag_device{};
        rl::environments::batch::cuda::observe_kernel<<<batch_cuda_blocks<SPEC>(), 32, 0, device.stream>>>(tag_device, batch, parameters, states, observation_type, observations, rng);
        check_status(device);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
