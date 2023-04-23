#ifndef MULTIROTOR_DYNAMICS_H
#define MULTIROTOR_DYNAMICS_H

#include "general_helper.h"
#include "quaternion_helper.h"

constexpr COUNTER_TYPE STATE_DIM = 13;
constexpr COUNTER_TYPE ACTION_DIM = 4;
#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

template <typename T, int N>
class Parameters {
public:
    T rotor_positions[N][3];
    T rotor_thrust_directions[N][3];
    T rotor_torque_directions[N][3];
    T thrust_constants[3];
    T torque_constant;
    T mass;
    T gravity[3];
    T J[3][3];
    T J_inv[3][3];
    T dt;
};

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(
        const Parameters<T, N>& params,

        // state
        const T state[STATE_DIM],

        // action
        const T rpms[N],

        T state_change[STATE_DIM]
        // state change
        ){
    const T* position_global_input        = &state[0];
    const T* orientation_global_input     = &state[3];
    const T* linear_velocity_global_input = &state[7];
    const T* angular_velocity_local_input = &state[10];

    T* linear_velocity_global     = &state_change[0];
    T* angular_velocity_global    = &state_change[3];
    T* linear_acceleration_global = &state_change[7];
    T* angular_acceleration_local = &state_change[10];

    T thrust[3];
    T torque[3];
    thrust[0] = 0; thrust[1] = 0; thrust[2] = 0;
    torque[0] = 0; torque[1] = 0; torque[2] = 0;
    // flops: N*23 => 4 * 23 = 92
    for(int i_rotor =0; i_rotor < N; i_rotor++){
        // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
        T rpm = rpms[i_rotor];
        T thrust_magnitude = params.thrust_constants[0] * rpm*rpm + params.thrust_constants[1] * rpm + params.thrust_constants[2];
        T rotor_thrust[3];
        scalar_multiply<T, 3>(params.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
        vector_add_accumulate<T, 3>(rotor_thrust, thrust);

        scalar_multiply_accumulate<T, 3>(params.rotor_torque_directions[i_rotor], thrust_magnitude * params.torque_constant, torque);
        cross_product_accumulate<T>(params.rotor_positions[i_rotor], rotor_thrust, torque);
    }

    // linear_velocity_global
    linear_velocity_global[0] = linear_velocity_global_input[0];
    linear_velocity_global[1] = linear_velocity_global_input[1];
    linear_velocity_global[2] = linear_velocity_global_input[2];

    // angular_velocity_global
    // flops: 16
    quaternion_derivative(orientation_global_input, angular_velocity_local_input, angular_velocity_global);

    // linear_acceleration_global
    // flops: 21
    rotate_vector_by_quaternion(orientation_global_input, thrust, linear_acceleration_global);
    // flops: 4
    scalar_multiply<T, 3>(linear_acceleration_global, 1/params.mass);
    vector_add_accumulate<T, 3>(params.gravity, linear_acceleration_global);

    T vector[3];
    T vector2[3];

    // angular_acceleration_local
    // flops: 9
    matrix_vector_product<T, 3, 3>(params.J, angular_velocity_local_input, vector);
    // flops: 6
    cross_product<T>(angular_velocity_local_input, vector, vector2);
    vector_sub<T, 3>(torque, vector2, vector);
    // flops: 9
    matrix_vector_product<T, 3, 3>(params.J_inv, vector, angular_acceleration_local);
    // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
}

template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void next_state_euler(const Parameters<T, N>& params, T state[STATE_DIM], T action[N], T dt, T next_state[STATE_DIM]){
    T dfdt[STATE_DIM];
    multirotor_dynamics<N>(params, state, action, dfdt);
    scalar_multiply<STATE_DIM>(dfdt, dt, next_state);
    vector_add_accumulate<STATE_DIM>(state, next_state);
}
template <typename T, int N>
BACKPROP_TOOLS_FUNCTION_PLACEMENT void next_state_rk4(const Parameters<T, N>& params, const T state[STATE_DIM], const T action[N], const T dt, T next_state[STATE_DIM]) {
    T* k1 = next_state; //[STATE_DIM];

    // flops: 157
    multirotor_dynamics<T, N>(params, state, action, k1);

    T var[STATE_DIM];

    // flops: 13
    scalar_multiply<T, STATE_DIM>(k1, dt/2, var);

    {
        T k2[STATE_DIM];
        vector_add_accumulate<T, STATE_DIM>(state, var);
        // flops: 157
        multirotor_dynamics<T, N>(params, var, action, k2);
        // flops: 13
        scalar_multiply<T, STATE_DIM>(k2, dt/2, var);
        // flops: 13
        scalar_multiply_accumulate<T, STATE_DIM>(k2, 2, k1);
    }
    {
        T k3[STATE_DIM];
        vector_add_accumulate<T, STATE_DIM>(state, var);
        // flops: 157
        multirotor_dynamics<T, N>(params, var, action, k3);
        // flops: 13
        scalar_multiply<T, STATE_DIM>(k3, dt, var);
        // flops: 13
        scalar_multiply_accumulate<T, STATE_DIM>(k3, 2, k1);
    }


    {
        T k4[STATE_DIM];
        vector_add_accumulate<T, STATE_DIM>(state, var);
        // flops: 157
        multirotor_dynamics<T, N>(params, var, action, k4);
        vector_add_accumulate<T, STATE_DIM>(k4, k1);
    }

    // flops: 13
    scalar_multiply<T, STATE_DIM>(k1, dt/6.0);
    vector_add_accumulate<T, STATE_DIM>(state, k1);
    // total flops: 157 + 13 + 157 + 13 + 13 + 157 + 13 + 13 + 157 + 13 = 706
}


#endif