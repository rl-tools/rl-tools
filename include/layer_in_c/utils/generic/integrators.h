#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif

namespace layer_in_c::utils::integrators{
    template<typename T, typename PARAMETER_TYPE, int STATE_DIM, int ACTION_DIM, auto DYNAMICS>
    FUNCTION_PLACEMENT void euler(const PARAMETER_TYPE& params, const T state[STATE_DIM], const T action[ACTION_DIM], const T dt, T next_state[STATE_DIM]) {
        T dfdt[STATE_DIM];
        DYNAMICS(params, state, action, dfdt);
        utils::vector_operations::scalar_multiply<STATE_DIM>(dfdt, dt, next_state);
        utils::vector_operations::add_accumulate<STATE_DIM>(state, next_state);
    }

    template<typename T, typename PARAMETER_TYPE, int STATE_DIM, int ACTION_DIM, auto DYNAMICS>
    FUNCTION_PLACEMENT void rk4(const PARAMETER_TYPE& params, const T state[STATE_DIM], const T action[ACTION_DIM], const T dt, T next_state[STATE_DIM]) {
        using namespace vector_operations;
        T *k1 = next_state; //[STATE_DIM];

        // flops: 157
        DYNAMICS(params, state, action, k1);

        T var[STATE_DIM];

        // flops: 13
        scalar_multiply<T, STATE_DIM>(k1, dt / 2, var);

        {
            T k2[STATE_DIM];
            add_accumulate<T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(params, var, action, k2);
            // flops: 13
            scalar_multiply<T, STATE_DIM>(k2, dt / 2, var);
            // flops: 13
            scalar_multiply_accumulate<T, STATE_DIM>(k2, 2, k1);
        }
        {
            T k3[STATE_DIM];
            add_accumulate<T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(params, var, action, k3);
            // flops: 13
            scalar_multiply<T, STATE_DIM>(k3, dt, var);
            // flops: 13
            scalar_multiply_accumulate<T, STATE_DIM>(k3, 2, k1);
        }


        {
            T k4[STATE_DIM];
            add_accumulate<T, STATE_DIM>(state, var);
            // flops: 157
            DYNAMICS(params, var, action, k4);
            add_accumulate<T, STATE_DIM>(k4, k1);
        }

        // flops: 13
        scalar_multiply<T, STATE_DIM>(k1, dt / 6.0);
        add_accumulate<T, STATE_DIM>(state, k1);
        // total flops: 157 + 13 + 157 + 13 + 13 + 157 + 13 + 13 + 157 + 13 = 706
    }
}
