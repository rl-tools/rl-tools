#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_BATCH_OPERATIONS_GENERIC_H

#include "environment.h"
#include "../../../random/operations_generic_array.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch){
        malloc(device, batch.environments);
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            malloc(device, get_ref(device, batch.environments, instance_i));
        }
    }

    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            free(device, get_ref(device, batch.environments, instance_i));
        }
        free(device, batch.environments);
    }

    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            init(device, get_ref(device, batch.environments, instance_i));
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            if(get(device, reset, instance_i)){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
                sample_initial_parameters(device, environment, parameter, rng_state);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            if(get(device, reset, instance_i)){
                auto& environment = get_ref(device, batch.environments, instance_i);
                auto& parameter = get_ref(device, parameters, instance_i);
                auto& state = get_ref(device, states, instance_i);
                auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
                sample_initial_state(device, environment, parameter, state, rng_state);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            auto& environment = get_ref(device, batch.environments, instance_i);
            auto& parameter = get_ref(device, parameters, instance_i);
            auto& state = get_ref(device, states, instance_i);
            auto& next_state = get_ref(device, next_states, instance_i);
            auto action = matrix_view(device, view(device, actions, instance_i));
            auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
            step(device, environment, parameter, state, action, next_state, rng_state);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            auto& environment = get_ref(device, batch.environments, instance_i);
            auto& parameter = get_ref(device, parameters, instance_i);
            auto& state = get_ref(device, states, instance_i);
            auto& next_state = get_ref(device, next_states, instance_i);
            auto action = matrix_view(device, view(device, actions, instance_i));
            auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
            set(device, rewards, reward(device, environment, parameter, state, action, next_state, rng_state), instance_i);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            auto& environment = get_ref(device, batch.environments, instance_i);
            auto& parameter = get_ref(device, parameters, instance_i);
            auto& state = get_ref(device, states, instance_i);
            auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
            set(device, terminated_flags, terminated(device, environment, parameter, state, rng_state), instance_i);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::batch::Independent<SPEC>& batch, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const OBSERVATION observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        for(typename SPEC::TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            auto& environment = get_ref(device, batch.environments, instance_i);
            auto& parameter = get_ref(device, parameters, instance_i);
            auto& state = get_ref(device, states, instance_i);
            auto observation_slice = view(device, observations, instance_i);
            auto observation = matrix_view(device, observation_slice);
            auto& rng_state = instance_rng<SPEC::INSTANCES>(rng, instance_i);
            observe(device, environment, parameter, state, observation_type, observation, rng_state);
        }
    }

    template <typename DEVICE, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC>
    void render(DEVICE&, rl::environments::batch::Independent<SPEC>&, Tensor<PARAMETER_SPEC>&, Tensor<STATE_SPEC>&, const Tensor<RESET_SPEC>&){ }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
