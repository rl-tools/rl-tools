#ifndef BACKPROP_TOOLS_UTILS_GENERIC_INTEGRATORS_H
#define BACKPROP_TOOLS_UTILS_GENERIC_INTEGRATORS_H

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

namespace backprop_tools::utils::integrators{
    template<typename T, typename PARAMETER_TYPE, auto STATE_DIM, auto ACTION_DIM, auto DYNAMICS>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void euler(const PARAMETER_TYPE& params, const T state[STATE_DIM], const T action[ACTION_DIM], const T dt, T next_state[STATE_DIM]) {
        T dfdt[STATE_DIM];
        DYNAMICS(params, state, action, dfdt);
        utils::vector_operations::scalar_multiply<STATE_DIM>(dfdt, dt, next_state);
        utils::vector_operations::add_accumulate<STATE_DIM>(state, next_state);
    }

    template<typename DEVICE, typename T, typename PARAMETER_TYPE, auto STATE_DIM, auto ACTION_DIM, auto DYNAMICS>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void rk4(DEVICE& device, const PARAMETER_TYPE& params, const T state[STATE_DIM], const T action[ACTION_DIM], const T dt, T next_state[STATE_DIM]) {
        using namespace vector_operations;
        T *k1 = next_state; //[STATE_DIM];

        // flops: 157
        DYNAMICS(device, params, state, action, k1);

        T var[STATE_DIM];

        // flops: 13
        scalar_multiply<DEVICE, T, STATE_DIM>(k1, dt / 2, var);

        {
            T k2[STATE_DIM];
            add_accumulate<DEVICE, T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k2);
            // flops: 13
            scalar_multiply<DEVICE, T, STATE_DIM>(k2, dt / 2, var);
            // flops: 13
            scalar_multiply_accumulate<DEVICE, T, STATE_DIM>(k2, 2, k1);
        }
        {
            T k3[STATE_DIM];
            add_accumulate<DEVICE, T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k3);
            // flops: 13
            scalar_multiply<DEVICE, T, STATE_DIM>(k3, dt, var);
            // flops: 13
            scalar_multiply_accumulate<DEVICE, T, STATE_DIM>(k3, 2, k1);
        }


        {
            T k4[STATE_DIM];
            add_accumulate<DEVICE, T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(device, params, var, action, k4);
            add_accumulate<DEVICE, T, STATE_DIM>(k4, k1);
        }

        // flops: 13
        scalar_multiply<DEVICE, T, STATE_DIM>(k1, dt / 6.0);
        add_accumulate<DEVICE, T, STATE_DIM>(state, k1);
        // total flops: 157 + 13 + 157 + 13 + 13 + 157 + 13 + 13 + 157 + 13 = 706
    }
}

#endif
