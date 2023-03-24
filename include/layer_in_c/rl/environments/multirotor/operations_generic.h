#ifndef LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H

#include "multirotor.h"
#include "parameters/reward_functions/reward_functions.h"

#include <layer_in_c/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <layer_in_c/utils/generic/integrators.h>
#include <layer_in_c/utils/generic/typing.h>

#ifndef LAYER_IN_C_FUNCTION_PLACEMENT
#define LAYER_IN_C_FUNCTION_PLACEMENT
#endif


namespace layer_in_c::rl::environments::multirotor {
    template<typename DEVICE, typename T, typename PARAMETERS, auto STATE_DIM, auto N, typename REWARD_FUNCTION>
    LAYER_IN_C_FUNCTION_PLACEMENT void multirotor_dynamics(
            DEVICE& device,
            const PARAMETERS& params,

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
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < N; i_rotor++) {
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm = rpms[i_rotor];
            T thrust_magnitude =
                    params.dynamics.thrust_constants[0] * rpm * rpm + params.dynamics.thrust_constants[1] * rpm +
                    params.dynamics.thrust_constants[2];
            T rotor_thrust[3];
            utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

            utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.torque_constant, torque);
            utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
        }

        // linear_velocity_global
        linear_velocity_global[0] = linear_velocity_global_input[0];
        linear_velocity_global[1] = linear_velocity_global_input[1];
        linear_velocity_global[2] = linear_velocity_global_input[2];

        // angular_velocity_global
        // flops: 16
        quaternion_derivative<DEVICE, T>(orientation_global_input, angular_velocity_local_input, angular_velocity_global);

        // linear_acceleration_global
        // flops: 21
        rotate_vector_by_quaternion<DEVICE, T>(orientation_global_input, thrust, linear_acceleration_global);
        // flops: 4
        utils::vector_operations::scalar_multiply<DEVICE, T, 3>(linear_acceleration_global, 1 / params.dynamics.mass);
        utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, linear_acceleration_global);

        T vector[3];
        T vector2[3];

        // angular_acceleration_local
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, angular_velocity_local_input, vector);
        // flops: 6
        utils::vector_operations::cross_product<DEVICE, T>(angular_velocity_local_input, vector, vector2);
        utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, angular_acceleration_local);
        // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
    }
}

namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::State& state){
        for(typename DEVICE::index_t i = 0; i < utils::typing::remove_reference<decltype(env)>::type::STATE_DIM; i++){
            state.state[i] = 0;
        }
        state.state[3] = 1;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    LAYER_IN_C_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        typename DEVICE::SPEC::MATH math_dev;
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        for(TI i = 0; i < 3; i++){
            state.state[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_position, env.parameters.mdp.init.max_position, rng);
        }
        // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
        if(env.parameters.mdp.init.max_angle > 0 && (random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) > env.parameters.mdp.init.guidance)){
            T u[3];
            for(TI i = 0; i < 3; i++){
                u[i] = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng);
            }
            state.state[3+0] = math::sqrt(math_dev, 1-u[0]) * math::sin(math_dev, 2*math::PI<T>*u[1]);
            state.state[3+1] = math::sqrt(math_dev, 1-u[0]) * math::cos(math_dev, 2*math::PI<T>*u[1]);
            state.state[3+2] = math::sqrt(math_dev,   u[0]) * math::sin(math_dev, 2*math::PI<T>*u[2]);
            state.state[3+3] = math::sqrt(math_dev,   u[0]) * math::cos(math_dev, 2*math::PI<T>*u[2]);
        }
        else{
            state.state[3+0] = 1;
            state.state[3+1] = 0;
            state.state[3+2] = 0;
            state.state[3+3] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.state[7+i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_linear_velocity, env.parameters.mdp.init.max_linear_velocity, rng);
        }
        for(TI i = 0; i < 3; i++){
            state.state[10+i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_angular_velocity, env.parameters.mdp.init.max_angular_velocity, rng);
        }
//        printf("initial state: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);
    }
    template<typename DEVICE, typename SPEC, typename OBS_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, Matrix<OBS_SPEC>& observation){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == ENVIRONMENT::STATE_DIM);
        for(typename DEVICE::index_t i = 0; i < ENVIRONMENT::STATE_DIM; i++){
            set(observation, 0, i, state.state[i]);
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Multirotor<SPEC>::State& next_state) {
        constexpr auto STATE_DIM = rl::environments::Multirotor<SPEC>::STATE_DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);
        typename SPEC::T action_scaled[ACTION_DIM];

        for(typename DEVICE::index_t action_i = 0; action_i < ACTION_DIM; action_i++){
            typename SPEC::T half_range = (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) / 2;
            action_scaled[action_i] = get(action, 0, action_i) * half_range + env.parameters.dynamics.action_limit.min + half_range;
        }
        utils::integrators::rk4<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION>>(device, env.parameters, state.state, action_scaled, env.parameters.integration.dt, next_state.state);
        typename SPEC::T quaternion_norm = 0;
        for(typename DEVICE::index_t state_i = 3; state_i < 3+4; state_i++){
            quaternion_norm += next_state.state[state_i] * next_state.state[state_i];
        }
        quaternion_norm = math::sqrt(typename DEVICE::SPEC::MATH(), quaternion_norm);
        for(typename DEVICE::index_t state_i = 3; state_i < 3+4; state_i++){
            next_state.state[state_i] /= quaternion_norm;
        }

        return env.parameters.integration.dt;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    LAYER_IN_C_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state) {
        return rl::environments::multirotor::parameters::reward_functions::reward(device, env, env.parameters.mdp.reward, state, action, next_state);
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    LAYER_IN_C_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        if(env.parameters.mdp.termination.enabled){
            for(typename DEVICE::index_t position_i = 0; position_i < 3; position_i++){
                if(math::abs(typename DEVICE::SPEC::MATH(), state.state[position_i]) > env.parameters.mdp.termination.position_threshold){
                    return true;
                }
            }
            for(typename DEVICE::index_t linear_velocity_i = 0; linear_velocity_i < 3; linear_velocity_i++){
                if(math::abs(typename DEVICE::SPEC::MATH(), state.state[3 + 4 + linear_velocity_i]) > env.parameters.mdp.termination.linear_velocity_threshold){
                    return true;
                }
            }
            for(typename DEVICE::index_t angular_velocity_i = 0; angular_velocity_i < 3; angular_velocity_i++){
                if(math::abs(typename DEVICE::SPEC::MATH(), state.state[3 + 4 + 3 + angular_velocity_i]) > env.parameters.mdp.termination.angular_velocity_threshold){
                    return true;
                }
            }
        }
        return false;
    }





}
#endif
