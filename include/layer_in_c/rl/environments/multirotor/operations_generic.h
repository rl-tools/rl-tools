#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include "general_helper.h"
#include "quaternion_helper.h"

#include <layer_in_c/utils/generic/integrators.h>

#ifndef FUNCTION_PLACEMENT
#define FUNCTION_PLACEMENT
#endif


namespace layer_in_c::rl::environments::multirotor {
    template<typename T, int N>
    FUNCTION_PLACEMENT void multirotor_dynamics(
            const Parameters<T, N> &params,

            // state
            const T state[STATE_DIM],

            // action
            const T rpms[N],

            T state_change[STATE_DIM]
            // state change
    ) {
        const T *position_global_input = &state[0];
        const T *orientation_global_input = &state[3];
        const T *linear_velocity_global_input = &state[7];
        const T *angular_velocity_local_input = &state[10];

        T *linear_velocity_global = &state_change[0];
        T *angular_velocity_global = &state_change[3];
        T *linear_acceleration_global = &state_change[7];
        T *angular_acceleration_local = &state_change[10];

        T thrust[3];
        T torque[3];
        thrust[0] = 0;
        thrust[1] = 0;
        thrust[2] = 0;
        torque[0] = 0;
        torque[1] = 0;
        torque[2] = 0;
        // flops: N*23 => 4 * 23 = 92
        for (int i_rotor = 0; i_rotor < N; i_rotor++) {
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm = rpms[i_rotor];
            T thrust_magnitude =
                    params.dynamics.thrust_constants[0] * rpm * rpm + params.dynamics.thrust_constants[1] * rpm +
                    params.dynamics.thrust_constants[2];
            T rotor_thrust[3];
            scalar_multiply<T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            vector_add_accumulate<T, 3>(rotor_thrust, thrust);

            scalar_multiply_accumulate<T, 3>(params.dynamics.rotor_torque_directions[i_rotor],
                                             thrust_magnitude * params.dynamics.torque_constant, torque);
            cross_product_accumulate<T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
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
        scalar_multiply<T, 3>(linear_acceleration_global, 1 / params.dynamics.mass);
        vector_add_accumulate<T, 3>(params.dynamics.gravity, linear_acceleration_global);

        T vector[3];
        T vector2[3];

        // angular_acceleration_local
        // flops: 9
        matrix_vector_product<T, 3, 3>(params.dynamics.J, angular_velocity_local_input, vector);
        // flops: 6
        cross_product<T>(angular_velocity_local_input, vector, vector2);
        vector_sub<T, 3>(torque, vector2, vector);
        // flops: 9
        matrix_vector_product<T, 3, 3>(params.dynamics.J_inv, vector, angular_acceleration_local);
        // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
    }
}

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    static typename SPEC::T step(const rl::environments::Multirotor<DEVICE, SPEC>& env, const rl::environments::multirotor::State<typename SPEC::T>& state, const typename SPEC::T action[rl::environments::multirotor::ACTION_DIM], rl::environments::multirotor::State<typename SPEC::T>& next_state) {
        typename SPEC::T action_scaled[rl::environments::multirotor::ACTION_DIM];
        for(size_t action_i = 0; action_i < rl::environments::multirotor::ACTION_DIM; action_i++){
            typename SPEC::T half_range = (env.parameters.action_limit.max - env.parameters.action_limit.min) / 2;
            action_scaled[action_i] = action[action_i] * half_range + env.parameters.action_limit.min + half_range;
        }
        utils::integrators::rk4<typename SPEC::T, typename std::remove_reference<decltype(env.parameters)>::type, rl::environments::multirotor::STATE_DIM, rl::environments::multirotor::ACTION_DIM, rl::environments::multirotor::multirotor_dynamics<typename SPEC::T, 4>>(env.parameters, state.state, action_scaled, env.parameters.dt, next_state.state);
        return env.parameters.dt;
    }
    template<typename DEVICE, typename SPEC>
    static typename SPEC::T reward(const rl::environments::Multirotor<DEVICE, SPEC>& env, const rl::environments::multirotor::State<typename SPEC::T>& state, const typename SPEC::T action[1], const rl::environments::multirotor::State<typename SPEC::T>& next_state){
        using T = typename SPEC::T;
        T acc = 0;
        for(size_t state_i = 0; state_i < rl::environments::multirotor::STATE_DIM; state_i++){
            if(state_i < 3){
                acc += state[state_i] * state[state_i] * env.parameters.reward.position;
            }
            else{
                if(state_i < 3+4){
                    acc += state[state_i] * state[state_i] * env.parameters.reward.orientation;
                }
                else{
                    if(state_i < 3+4+3){
                        acc += state[state_i] * state[state_i] * env.parameters.reward.linear_velocity;
                    }
                    else{
                        acc += state[state_i] * state[state_i] * env.parameters.reward.angular_velocity;
                    }
                }
            }
        }
        return std::exp(-acc);
    }

    template<typename DEVICE, typename SPEC>
    static bool terminated(const rl::environments::Multirotor<DEVICE, SPEC>& env, const typename rl::environments::multirotor::State<typename SPEC::T> state){
        return false;
    }
}
#endif
