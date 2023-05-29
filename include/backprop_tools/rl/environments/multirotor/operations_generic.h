#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include <backprop_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <backprop_tools/utils/generic/integrators.h>
#include <backprop_tools/utils/generic/typing.h>

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif


namespace backprop_tools::rl::environments::multirotor {
    template<typename DEVICE, typename T, typename PARAMETERS, auto STATE_DIM, auto N, typename REWARD_FUNCTION>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(
            DEVICE& device,
            const PARAMETERS& params,

            // state
            const T state[STATE_DIM],

            // action
            const T action[N],

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
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < N; i_rotor++){
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm;
            if constexpr(STATE_DIM == 13){
                rpm = action[i_rotor];
            }
            else{
                T *rpm_change_rate = &state_change[13];
                const T *rpms = &state[13];
                rpm = rpms[i_rotor];
                rpm_change_rate[i_rotor] = (action[i_rotor] - rpm) * 1/params.dynamics.rpm_time_constant;
            }
            T thrust_magnitude = params.dynamics.thrust_constants[0] + params.dynamics.thrust_constants[1] * rpm + params.dynamics.thrust_constants[2] * rpm * rpm;
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

namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE, rl::environments::Multirotor<SPEC>){

    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::State& state){
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        for(typename DEVICE::index_t i = 0; i < STATE::DIM; i++){
            state.state[i] = 0;
        }
        state.state[3] = 1;
        if constexpr(STATE::DIM == 17){
            for(typename DEVICE::index_t i = 0; i < 4; i++){
                state.state[13 + i] = env.parameters.dynamics.action_limit.min;
            }
        }
        env.current_dynamics = env.parameters.dynamics;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        typename DEVICE::SPEC::MATH math_dev;
        typename DEVICE::SPEC::RANDOM random_dev;
        using MULTIROTOR_TYPE = rl::environments::Multirotor<SPEC>;
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
        if constexpr(MULTIROTOR_TYPE::State::DIM == 17){
            for(TI i = 0; i < 4; i++){
                state.state[13+i] = random::uniform_real_distribution(random_dev, env.parameters.dynamics.action_limit.min, env.parameters.dynamics.action_limit.max, rng);
            }
        }
        env.current_dynamics = env.parameters.dynamics;
        T J_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)5, rng);
        env.current_dynamics.J[0][0] *= J_factor;
        env.current_dynamics.J[1][1] *= J_factor;
        env.current_dynamics.J[2][2] *= J_factor;
        env.current_dynamics.J_inv[0][0] /= J_factor;
        env.current_dynamics.J_inv[1][1] /= J_factor;
        env.current_dynamics.J_inv[2][2] /= J_factor;
        T mass_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)2, rng);
        env.current_dynamics.mass *= mass_factor;
//        printf("initial state: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);
    }
    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        using STATE = typename ENVIRONMENT::State;
        using T = typename SPEC::T;
        constexpr T position_noise_std = 0.003;
        constexpr T orientation_noise_std = 0.001;
        constexpr T linear_velocity_noise = 0.01;
        constexpr T angular_velocity_noise = 0.003;
        static_assert(OBS_SPEC::ROWS == 1);
//        add_scalar(device, device.logger, "quaternion_w", state.state[3], 1000);
        if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::Normal){
            static_assert(OBS_SPEC::COLS == 13);
            for(typename DEVICE::index_t i = 0; i < 13; i++){
                set(observation, 0, i, state.state[i]);
            }
            if constexpr(SPEC::STATIC_PARAMETERS::ENFORCE_POSITIVE_QUATERNION){
                if(get(observation, 0, 3) < 0){
                    for(typename DEVICE::index_t observation_i = 3; observation_i < 7; observation_i++){
                        set(observation, 0, observation_i, -get(observation, 0, observation_i));
                    }
                }
            }
            else{
                if constexpr(SPEC::STATIC_PARAMETERS::RANDOMIZE_QUATERNION_SIGN){
                    if(random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), 0, 1, rng) == 0){
                        for(typename DEVICE::index_t observation_i = 3; observation_i < 7; observation_i++){
                            set(observation, 0, observation_i, -get(observation, 0, observation_i));
                        }
                    }
                }
            }
            for(typename DEVICE::index_t i = 0; i < 3; i++){
                increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, position_noise_std, rng));
            }
            for(typename DEVICE::index_t i = 3; i < 7; i++){
                increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, orientation_noise_std, rng));
            }
            for(typename DEVICE::index_t i = 7; i < 10; i++){
                increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, linear_velocity_noise, rng));
            }
            for(typename DEVICE::index_t i = 10; i < 13; i++){
                increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, angular_velocity_noise, rng));
            }

        }
        else{
            if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::DoubleQuaternion){
                static_assert(OBS_SPEC::COLS == 17);
                for(typename DEVICE::index_t i = 0; i < 3; i++){
                    set(observation, 0, i, state.state[i]);
                }
                typename SPEC::T sign = state.state[3] > 0 ? 1 : -1;
                for(typename DEVICE::index_t i = 3; i < 7; i++){
//                    set(observation, 0, i+0,   state.state[i]);
                    set(observation, 0, i+0,   sign * state.state[i]);
                    set(observation, 0, i+4, - sign * state.state[i]);
//                    set(observation, 0, i+4, 0);
                }
                for(typename DEVICE::index_t i = 7; i < 13; i++){
                    set(observation, 0, i+4, state.state[i]);
                }
            }
            else{
                if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::RotationMatrix){
                    static_assert(OBS_SPEC::COL_PITCH == 1); // so that we can use the quaternion_to_rotation_matrix function
                    for(typename DEVICE::index_t i = 0; i < 3; i++){
                        set(observation, 0, i, state.state[i]);
                    }
                    const typename SPEC::T* q = &state.state[3];
                    set(observation, 0, 3 + 0, (1 - 2*q[2]*q[2] - 2*q[3]*q[3]));
                    set(observation, 0, 3 + 1, (    2*q[1]*q[2] - 2*q[0]*q[3]));
                    set(observation, 0, 3 + 2, (    2*q[1]*q[3] + 2*q[0]*q[2]));
                    set(observation, 0, 3 + 3, (    2*q[1]*q[2] + 2*q[0]*q[3]));
                    set(observation, 0, 3 + 4, (1 - 2*q[1]*q[1] - 2*q[3]*q[3]));
                    set(observation, 0, 3 + 5, (    2*q[2]*q[3] - 2*q[0]*q[1]));
                    set(observation, 0, 3 + 6, (    2*q[1]*q[3] - 2*q[0]*q[2]));
                    set(observation, 0, 3 + 7, (    2*q[2]*q[3] + 2*q[0]*q[1]));
                    set(observation, 0, 3 + 8, (1 - 2*q[1]*q[1] - 2*q[2]*q[2]));
                    for(typename DEVICE::index_t i = 7; i < 13; i++){
                        set(observation, 0, i-4+9, state.state[i]);
                    }

                    for(typename DEVICE::index_t i = 0; i < 3; i++){
                        increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, position_noise_std, rng));
                    }
                    for(typename DEVICE::index_t i = 3; i < 12; i++){
                        increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, orientation_noise_std, rng));
                    }
                    for(typename DEVICE::index_t i = 12; i < 15; i++){
                        increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, linear_velocity_noise, rng));
                    }
                    for(typename DEVICE::index_t i = 15; i < 18; i++){
                        increment(observation, 0, i, random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, angular_velocity_noise, rng));
                    }
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Multirotor<SPEC>::State& next_state) {
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        constexpr auto STATE_DIM = STATE::DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);
        typename SPEC::T action_scaled[ACTION_DIM];

        for(typename DEVICE::index_t action_i = 0; action_i < ACTION_DIM; action_i++){
            typename SPEC::T half_range = (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) / 2;
            action_scaled[action_i] = get(action, 0, action_i) * half_range + env.parameters.dynamics.action_limit.min + half_range;
        }
        utils::integrators::rk4<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION>>(device, env.parameters, state.state, action_scaled, env.parameters.integration.dt, next_state.state);
//        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE_DIM, ACTION_DIM, typename SPEC::PARAMETERS::MDP::REWARD_FUNCTION>>(device, env.parameters, state.state, action_scaled, env.parameters.integration.dt, next_state.state);
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

    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
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
#include "parameters/reward_functions/reward_functions.h"
namespace backprop_tools{
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        return rl::environments::multirotor::parameters::reward_functions::reward(device, env, env.parameters.mdp.reward, state, action, next_state, rng);
    }
}

#endif
