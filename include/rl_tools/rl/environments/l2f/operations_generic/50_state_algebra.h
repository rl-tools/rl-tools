#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_STATE_ALGEBRA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_L2F_OPERATIONS_GENERIC_STATE_ALGEBRA_H

#include "../multirotor.h"

#include <rl_tools/utils/generic/vector_operations.h>
#include "../quaternion_helper.h"

#include <rl_tools/utils/generic/typing.h>

#include <rl_tools/rl/environments/operations_generic.h>

// This file contains algebraic operations for states that REQUIRE_INTEGRATION.

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::l2f{
    // State arithmetic for RK4 integration
    // scalar multiply
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StateBase<STATE_SPEC>& state, T scalar, rl::environments::l2f::StateBase<STATE_SPEC>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = scalar * state.position[i]        ;
            out.orientation[i]      = scalar * state.orientation[i]     ;
            out.linear_velocity[i]  = scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] = scalar * state.angular_velocity[i];
        }
        out.orientation[3] = scalar * state.orientation[3];
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& state, T scalar, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& out){
        scalar_multiply(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        out.position_integral = scalar * out.position_integral;
        out.orientation_integral = scalar * out.orientation_integral;
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const typename rl::environments::l2f::StateRotors<STATE_SPEC>& state, T scalar, rl::environments::l2f::StateRotors<STATE_SPEC>& out){
        scalar_multiply(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        if constexpr(!STATE_SPEC::CLOSED_FORM){
            for(int i = 0; i < 4; ++i){
                out.rpm[i] = scalar * state.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, const STATE& state, T scalar, STATE& out, rl_tools::utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, static_cast<const typename STATE::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
    // scalar multiply in place
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StateBase<STATE_SPEC>& state, T scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& state, T scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, typename rl::environments::l2f::StateRotors<STATE_SPEC>& state, T scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply(DEVICE& device, STATE& state, T scalar, rl_tools::utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply(device, static_cast<typename STATE::NEXT_COMPONENT&>(state), scalar);
    }

    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StateBase<STATE_SPEC>& state, T scalar, rl::environments::l2f::StateBase<STATE_SPEC>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         += scalar * state.position[i]        ;
            out.orientation[i]      += scalar * state.orientation[i]     ;
            out.linear_velocity[i]  += scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] += scalar * state.angular_velocity[i];
        }
        out.orientation[3] += scalar * state.orientation[3];
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& state, T scalar, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& out){
        scalar_multiply_accumulate(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        out.position_integral += scalar * out.position_integral;
        out.orientation_integral += scalar * out.orientation_integral;
    }
    template<typename DEVICE, typename STATE_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::l2f::StateRotors<STATE_SPEC>& state, T scalar, rl::environments::l2f::StateRotors<STATE_SPEC>& out){
        scalar_multiply_accumulate(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        if constexpr(!STATE_SPEC::CLOSED_FORM) {
            for(int i = 0; i < 4; ++i){
                out.rpm[i] += scalar * state.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT static void scalar_multiply_accumulate(DEVICE& device, const STATE& state, T scalar, STATE& out, rl_tools::utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        scalar_multiply_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(state), scalar, static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }

    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateBase<STATE_SPEC>& s1, const rl::environments::l2f::StateBase<STATE_SPEC>& s2, rl::environments::l2f::StateBase<STATE_SPEC>& out){
        for(int i = 0; i < 3; ++i){
            out.position[i]         = s1.position[i] + s2.position[i];
            out.orientation[i]      = s1.orientation[i] + s2.orientation[i];
            out.linear_velocity[i]  = s1.linear_velocity[i] + s2.linear_velocity[i];
            out.angular_velocity[i] = s1.angular_velocity[i] + s2.angular_velocity[i];
        }
        out.orientation[3] = s1.orientation[3] + s2.orientation[3];
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& s1, const rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& s2, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& out){
        add_accumulate(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(s1), static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(s2), static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        out.position_integral = s1.position_integral + s2.position_integral;
        out.orientation_integral = s1.orientation_integral + s2.orientation_integral;
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const typename rl::environments::l2f::StateRotors<STATE_SPEC>& s1, const rl::environments::l2f::StateRotors<STATE_SPEC>& s2, rl::environments::l2f::StateRotors<STATE_SPEC>& out){
        add_accumulate(device, static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(s1), static_cast<const typename STATE_SPEC::NEXT_COMPONENT&>(s2), static_cast<typename STATE_SPEC::NEXT_COMPONENT&>(out));
        if constexpr(!STATE_SPEC::CLOSED_FORM) {
            for(int i = 0; i < 4; ++i){
                out.rpm[i] = s1.rpm[i] + s2.rpm[i];
            }
        }
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s1, const STATE& s2, STATE& out, rl_tools::utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s1), static_cast<const typename STATE::NEXT_COMPONENT&>(s2), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const rl::environments::l2f::StateBase<STATE_SPEC>& s, rl::environments::l2f::StateBase<STATE_SPEC>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& s, rl::environments::l2f::StatePoseErrorIntegral<STATE_SPEC>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename STATE_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const rl::environments::l2f::StateRotors<STATE_SPEC>& s, rl::environments::l2f::StateRotors<STATE_SPEC>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename STATE>
    RL_TOOLS_FUNCTION_PLACEMENT static void add_accumulate(DEVICE& device, const STATE& s, STATE& out, rl_tools::utils::typing::enable_if_t<!STATE::REQUIRES_INTEGRATION, bool> disable = false){
        static_assert(!STATE::REQUIRES_INTEGRATION);
        add_accumulate(device, static_cast<const typename STATE::NEXT_COMPONENT&>(s), static_cast<const typename STATE::NEXT_COMPONENT&>(out), static_cast<typename STATE::NEXT_COMPONENT&>(out));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif