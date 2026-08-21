#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_OPERATIONS_GENERIC_H

#include "multi_environment.h"
#include "../operations_generic_batch.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::multi_environment{
        template <typename MULTI_ENVIRONMENT, typename SPEC>
        constexpr typename MULTI_ENVIRONMENT::TI instances_per_environment(){
            constexpr auto TOTAL = get<0>(typename SPEC::SHAPE{});
            constexpr auto N = MULTI_ENVIRONMENT::NUMBER_OF_ENVIRONMENTS;
            static_assert(TOTAL % N == 0, "instance tensors must split into contiguous equal blocks across the members");
            static_assert(MULTI_ENVIRONMENT::INSTANCES_PER_ENVIRONMENT == 0 || TOTAL / N == MULTI_ENVIRONMENT::INSTANCES_PER_ENVIRONMENT);
            return TOTAL / N;
        }
    }

    // members that declare a SharedContext own shared resources and provide their own
    // composite lifecycle overloads
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename utils::typing::enable_if<utils::typing::is_same_v<typename rl::environments::multi_environment::SharedContext<ENVIRONMENT>::TYPE, rl::environments::multi_environment::EmptySharedContext>, bool>::type = true>
    void malloc(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env){
        using TI = typename ENVIRONMENT::TI;
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            malloc(device, env.environments[environment_i]);
        }
    }
    // members that declare a SharedContext own shared resources and provide their own
    // composite lifecycle overloads
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename utils::typing::enable_if<utils::typing::is_same_v<typename rl::environments::multi_environment::SharedContext<ENVIRONMENT>::TYPE, rl::environments::multi_environment::EmptySharedContext>, bool>::type = true>
    void free(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env){
        using TI = typename ENVIRONMENT::TI;
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            free(device, env.environments[environment_i]);
        }
    }
    // members that declare a SharedContext own shared resources and provide their own
    // composite lifecycle overloads
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename utils::typing::enable_if<utils::typing::is_same_v<typename rl::environments::multi_environment::SharedContext<ENVIRONMENT>::TYPE, rl::environments::multi_environment::EmptySharedContext>, bool>::type = true>
    void init(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env){
        using TI = typename ENVIRONMENT::TI;
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            init(device, env.environments[environment_i]);
        }
    }

    namespace rl::environments::multi_environment{
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void _sample_initial_parameters(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, PARAMETER_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            sample_initial_parameters(device, env.environments[environment_i], parameters_block, reset_block, rng);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void _sample_initial_state(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto reset_block = view_range(device, reset_mask, environment_i * M, tensor::ViewSpec<0, M>{});
            sample_initial_state(device, env.environments[environment_i], parameters_block, states_block, reset_block, rng);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void _observe(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto observations_block = view_range(device, observations, environment_i * M, tensor::ViewSpec<0, M>{});
            observe(device, env.environments[environment_i], parameters_block, states_block, observation_type, observations_block, rng);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void _step(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto actions_block = view_range(device, actions, environment_i * M, tensor::ViewSpec<0, M>{});
            auto next_states_block = view_range(device, next_states, environment_i * M, tensor::ViewSpec<0, M>{});
            step(device, env.environments[environment_i], parameters_block, states_block, actions_block, next_states_block, rng);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void _reward(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto actions_block = view_range(device, actions, environment_i * M, tensor::ViewSpec<0, M>{});
            auto next_states_block = view_range(device, next_states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto rewards_block = view_range(device, rewards, environment_i * M, tensor::ViewSpec<0, M>{});
            reward(device, env.environments[environment_i], parameters_block, states_block, actions_block, next_states_block, rewards_block, rng);
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void _terminated(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using MULTI_ENVIRONMENT = rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>;
        using TI = typename MULTI_ENVIRONMENT::TI;
        constexpr TI M = rl::environments::multi_environment::instances_per_environment<MULTI_ENVIRONMENT, STATE_SPEC>();
        for(TI environment_i = 0; environment_i < NUMBER_OF_ENVIRONMENTS; environment_i++){
            auto parameters_block = view_range(device, parameters, environment_i * M, tensor::ViewSpec<0, M>{});
            auto states_block = view_range(device, states, environment_i * M, tensor::ViewSpec<0, M>{});
            auto terminated_block = view_range(device, terminated_flags, environment_i * M, tensor::ViewSpec<0, M>{});
            terminated(device, env.environments[environment_i], parameters_block, states_block, terminated_block, rng);
        }
    }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        rl::environments::multi_environment::_sample_initial_parameters(device, env, parameters, reset_mask, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        rl::environments::multi_environment::_sample_initial_state(device, env, parameters, states, reset_mask, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG>
    void observe(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        rl::environments::multi_environment::_observe(device, env, parameters, states, observation_type, observations, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG>
    void step(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        rl::environments::multi_environment::_step(device, env, parameters, states, actions, next_states, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG>
    void reward(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        rl::environments::multi_environment::_reward(device, env, parameters, states, actions, next_states, rewards, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename ENVIRONMENT::TI NUMBER_OF_ENVIRONMENTS, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG>
    void terminated(DEVICE& device, rl::environments::MultiEnvironment<ENVIRONMENT, NUMBER_OF_ENVIRONMENTS>& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        rl::environments::multi_environment::_terminated(device, env, parameters, states, terminated_flags, rng);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
