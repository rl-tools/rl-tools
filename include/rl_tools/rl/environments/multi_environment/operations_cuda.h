#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_OPERATIONS_CUDA_H

#include "operations_generic.h"
#include "../operations_cuda_batch.h"

// exact-match entry points so a (CUDA device, MultiEnvironment) call is unambiguous between the
// generic fan-out and the kernel-mapped defaults: the composite always fans out to its members
// (whose verbs then resolve to their own CUDA overloads)

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        rl::environments::multi_environment::_sample_initial_parameters(device, env, parameters, reset_mask, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        rl::environments::multi_environment::_sample_initial_state(device, env, parameters, states, reset_mask, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void observe(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        rl::environments::multi_environment::_observe(device, env, parameters, states, observation_type, observations, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        rl::environments::multi_environment::_step(device, env, parameters, states, actions, next_states, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        rl::environments::multi_environment::_reward(device, env, parameters, states, actions, next_states, rewards, rng);
    }
    template <typename DEV_SPEC, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(devices::CUDA<DEV_SPEC>& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        rl::environments::multi_environment::_terminated(device, env, parameters, states, terminated_flags, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
