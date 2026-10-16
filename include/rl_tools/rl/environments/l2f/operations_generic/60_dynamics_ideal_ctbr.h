#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_IDEAL_CTBR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_DYNAMICS_IDEAL_CTBR_H

#include "../multirotor.h"
#include "../quaternion_helper.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename PARAMETERS, typename TI, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT T clamp_ideal_ctbr_action(DEVICE& device, const rl::environments::Multirotor<SPEC>&, const PARAMETERS& parameters, TI index, T value){
        constexpr auto limits = SPEC::STATIC_PARAMETERS::IDEAL_CTBR;
        if(index == 0){
            T maximum = limits.thrust_max;
            if constexpr(limits.thrust_max == 0){
                T command = parameters.dynamics.action_limit.max;
                for(TI rotor = 0; rotor < 4; rotor++){
                    const auto& coefficients = parameters.dynamics.rotor_thrust_coefficients[rotor];
                    maximum += parameters.dynamics.rotor_thrust_directions[rotor][2] * (coefficients[0] + coefficients[1] * command + coefficients[2] * command * command);
                }
            }
            return math::clamp(device.math, value, (T)0, maximum);
        }
        return math::clamp(device.math, value, -limits.rate_limit[index - 1], limits.rate_limit[index - 1]);
    }

    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics(DEVICE&, const PARAMETERS&, const STATE&, const T*, STATE&);

    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const rl::environments::l2f::StateBase<STATE_SPEC>& state, const T* action, rl::environments::l2f::StateBase<STATE_SPEC>& change){
        T thrust[3] = {0, 0, action[0]};
        rl::environments::l2f::quaternion_derivative<DEVICE, T>(state.orientation, action + 1, change.orientation);
        rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, change.linear_velocity);
        for(typename DEVICE::index_t axis = 0; axis < 3; axis++){
            change.position[axis] = state.linear_velocity[axis];
            change.linear_velocity[axis] = change.linear_velocity[axis] / parameters.dynamics.mass + parameters.dynamics.gravity[axis];
            change.angular_velocity[axis] = 0;
        }
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const rl::environments::l2f::StateRandomForce<STATE_SPEC>& state, const T* action, rl::environments::l2f::StateRandomForce<STATE_SPEC>& change){
        ideal_ctbr_dynamics(device, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(change));
        for(typename DEVICE::index_t axis = 0; axis < 3; axis++){
            change.linear_velocity[axis] += state.force[axis] / parameters.dynamics.mass;
        }
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const rl::environments::l2f::StateRotors<STATE_SPEC>& state, const T* action, rl::environments::l2f::StateRotors<STATE_SPEC>& change){
        ideal_ctbr_dynamics(device, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(change));
        if constexpr(!STATE_SPEC::CLOSED_FORM){
            for(typename DEVICE::index_t rotor = 0; rotor < 4; rotor++){
                change.rpm[rotor] = 0;
            }
        }
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& state, const T* action, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& change){
        ideal_ctbr_dynamics(device, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(change));
        for(typename DEVICE::index_t axis = 0; axis < 3; axis++){
            change.position_integral[axis] = state.position[axis];
        }
        change.orientation_integral = 2 * math::acos(device.math, math::clamp(device.math, state.orientation[0], (T)-1, (T)1));
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const rl::environments::l2f::StateLinearAcceleration<STATE_SPEC>& state, const T* action, rl::environments::l2f::StateLinearAcceleration<STATE_SPEC>& change){
        ideal_ctbr_dynamics(device, parameters, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), action, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(change));
    }
    template<typename DEVICE, typename PARAMETERS, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics_component(DEVICE& device, const PARAMETERS& parameters, const STATE& state, const T* action, STATE& change){
        static_assert(!STATE::REQUIRES_INTEGRATION, "Integrated state component needs ideal CTBR dynamics");
        ideal_ctbr_dynamics(device, parameters, static_cast<const typename STATE::NEXT_COMPONENT&>(state), action, static_cast<typename STATE::NEXT_COMPONENT&>(change));
    }
    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT void ideal_ctbr_dynamics(DEVICE& device, const PARAMETERS& parameters, const STATE& state, const T* action, STATE& change){
        ideal_ctbr_dynamics_component(device, parameters, state, action, change);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
