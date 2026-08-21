#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_BATCH_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_OPERATIONS_GENERIC_BATCH_H

#include "operations_generic.h"

// The batch tier of the environment contract: the per-instance verbs, overloaded
// per-environment-over-instances — one environment (the shared object) plus tensors of instance
// data with a shared leading instance dimension. These generic defaults map the per-instance
// operations exactly like the runners' internal loops; batch-native environments provide
// more-specialized overloads of the same verbs.

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::environments::batch{
        // batch-native environments (BATCH_NATIVE = true, inherited by their wrappers) provide
        // their own batch overloads; the generic mapped defaults must not capture them (an exact
        // generic match would otherwise outrank the environment's derived-to-base overloads)
        template <typename ENVIRONMENT, typename = void>
        struct BatchNative{
            static constexpr bool VALUE = false;
        };
        template <typename ENVIRONMENT>
        struct BatchNative<ENVIRONMENT, utils::typing::void_t<decltype(ENVIRONMENT::BATCH_NATIVE)>>{
            static constexpr bool VALUE = ENVIRONMENT::BATCH_NATIVE;
        };
        template <typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC>
        constexpr bool check_instance_tensors(){
            static_assert(utils::typing::is_same_v<typename PARAMETER_SPEC::T, typename ENVIRONMENT::Parameters>);
            static_assert(utils::typing::is_same_v<typename STATE_SPEC::T, typename ENVIRONMENT::State>);
            static_assert(length(typename PARAMETER_SPEC::SHAPE{}) == 1);
            static_assert(length(typename STATE_SPEC::SHAPE{}) == 1);
            static_assert(get<0>(typename PARAMETER_SPEC::SHAPE{}) == get<0>(typename STATE_SPEC::SHAPE{}));
            return true;
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename RESET_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void sample_initial_parameters(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(utils::typing::is_same_v<typename PARAMETER_SPEC::T, typename ENVIRONMENT::Parameters>);
        static_assert(utils::typing::is_same_v<typename RESET_SPEC::T, bool>);
        constexpr TI INSTANCES = get<0>(typename PARAMETER_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename RESET_SPEC::SHAPE{}));
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            if(get(device, reset_mask, instance_i)){
                sample_initial_parameters(device, env, get_ref(device, parameters, instance_i), rng);
            }
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename RESET_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void sample_initial_state(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<RESET_SPEC>& reset_mask, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(utils::typing::is_same_v<typename RESET_SPEC::T, bool>);
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename RESET_SPEC::SHAPE{}));
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            if(get(device, reset_mask, instance_i)){
                sample_initial_state(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng);
            }
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename OBSERVATION_TYPE, typename OBSERVATION_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void observe(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, OBSERVATION_TYPE observation_type, Tensor<OBSERVATION_SPEC>& observations, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename OBSERVATION_SPEC::SHAPE{}));
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            auto observation_slice = view(device, observations, instance_i);
            auto observation_matrix = matrix_view(device, observation_slice);
            observe(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), observation_type, observation_matrix, rng);
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void step(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(utils::typing::is_same_v<typename NEXT_STATE_SPEC::T, typename ENVIRONMENT::State>);
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename ACTION_SPEC::SHAPE{}));
        static_assert(INSTANCES == get<0>(typename NEXT_STATE_SPEC::SHAPE{}));
        static_assert(get<1>(typename ACTION_SPEC::SHAPE{}) == ENVIRONMENT::ACTION_DIM);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action_matrix;
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
            }
            step(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng);
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename ACTION_SPEC, typename NEXT_STATE_SPEC, typename REWARD_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void reward(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<ACTION_SPEC>& actions, Tensor<NEXT_STATE_SPEC>& next_states, Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename REWARD_SPEC::SHAPE{}));
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            Matrix<matrix::Specification<typename ACTION_SPEC::T, TI, 1, ENVIRONMENT::ACTION_DIM, false>> action_matrix;
            for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
                set(action_matrix, 0, action_i, get(device, actions, instance_i, action_i));
            }
            set(device, rewards, reward(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), action_matrix, get_ref(device, next_states, instance_i), rng), instance_i);
        }
    }

    template <typename DEVICE, typename ENVIRONMENT, typename PARAMETER_SPEC, typename STATE_SPEC, typename TERMINATED_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA && !rl::environments::batch::BatchNative<ENVIRONMENT>::VALUE, bool>::type = true>
    void terminated(DEVICE& device, ENVIRONMENT& env, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, Tensor<TERMINATED_SPEC>& terminated_flags, RNG& rng){
        using TI = typename DEVICE::index_t;
        static_assert(rl::environments::batch::check_instance_tensors<ENVIRONMENT, PARAMETER_SPEC, STATE_SPEC>());
        static_assert(utils::typing::is_same_v<typename TERMINATED_SPEC::T, bool>);
        constexpr TI INSTANCES = get<0>(typename STATE_SPEC::SHAPE{});
        static_assert(INSTANCES == get<0>(typename TERMINATED_SPEC::SHAPE{}));
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            set(device, terminated_flags, terminated(device, env, get_ref(device, parameters, instance_i), get_ref(device, states, instance_i), rng), instance_i);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
