#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_GENERIC_INTEGRATORS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_GENERIC_INTEGRATORS_H

#ifndef RL_TOOLS_FUNCTION_PLACEMENT
#define RL_TOOLS_FUNCTION_PLACEMENT
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::integrators{
    template<typename DEVICE, typename T, typename PARAMETER_TYPE, typename STATE, auto ACTION_DIM, auto DYNAMICS>
    RL_TOOLS_FUNCTION_PLACEMENT void euler(DEVICE& device, const PARAMETER_TYPE& params, const STATE& state, const T action[ACTION_DIM], const T dt, STATE& next_state) {
        DYNAMICS(device, params, state, action, next_state);
        scalar_multiply(device, next_state, dt);
        add_accumulate(device, state, next_state);
    }

    template<typename DEVICE, typename T, typename PARAMETER_TYPE, typename STATE, auto ACTION_DIM, auto DYNAMICS>
    RL_TOOLS_FUNCTION_PLACEMENT void rk4(DEVICE& device, const PARAMETER_TYPE& params, const STATE& state, const T action[ACTION_DIM], const T dt, STATE& next_state) {
        next_state = state;
        STATE& k1 = next_state; //[STATE_DIM];

        // flops: 157
        DYNAMICS(device, params, state, action, k1);

        STATE var = state;

        // flops: 13
        scalar_multiply(device, k1, dt / 2, var);

        {
            STATE k2 = state;
            add_accumulate(device, state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k2);
            // flops: 13
            scalar_multiply(device, k2, dt / 2, var);
            // flops: 13
            scalar_multiply_accumulate(device, k2, 2, k1);
        }
        {
            STATE k3 = state;
            add_accumulate(device, state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k3);
            // flops: 13
            scalar_multiply(device, k3, dt, var);
            // flops: 13
            scalar_multiply_accumulate(device, k3, 2, k1);
        }


        {
            STATE k4 = state;
            add_accumulate(device, state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k4);
            add_accumulate(device, k4, k1);
        }

        // flops: 13
        scalar_multiply(device, k1, dt / 6.0);
        add_accumulate(device, state, k1);
        // total flops: 157 + 13 + 157 + 13 + 13 + 157 + 13 + 13 + 157 + 13 = 706
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
