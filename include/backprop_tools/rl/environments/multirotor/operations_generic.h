#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_OPERATIONS_GENERIC_H

#include "multirotor.h"

#include <backprop_tools/utils/generic/vector_operations.h>
#include "quaternion_helper.h"

#include <backprop_tools/utils/generic/typing.h>

#ifndef BACKPROP_TOOLS_FUNCTION_PLACEMENT
#define BACKPROP_TOOLS_FUNCTION_PLACEMENT
#endif

namespace backprop_tools{
    // State arithmetic for RK4 integration
    template<typename DEVICE, typename T, typename TI, typename T2>
    static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateLatentEmpty<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateLatentEmpty<T, TI>& out){ }
    template<typename DEVICE, typename T, typename TI, typename T2>
    static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& out){
        out.force[0] = scalar * state.force[0];
        out.force[1] = scalar * state.force[1];
        out.force[2] = scalar * state.force[2];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, T2 scalar, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& out){
        scalar_multiply(device, (const LATENT_STATE&)state, scalar, (LATENT_STATE&)out);
        for(int i = 0; i < 3; ++i){
            out.position[i]         = scalar * state.position[i]        ;
            out.orientation[i]      = scalar * state.orientation[i]     ;
            out.linear_velocity[i]  = scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] = scalar * state.angular_velocity[i];
        }
        out.orientation[3] = scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply(DEVICE& device, const typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, T2 scalar, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& out){
        scalar_multiply(device, static_cast<const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(state), scalar, static_cast<typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(out));
        for(int i = 0; i < 4; ++i){
            out.rpm[i] = scalar * state.rpm[i];
        }
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply(DEVICE& device, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, T2 scalar){
        scalar_multiply(device, state, scalar, state);
    }
    template<typename DEVICE, typename T, typename TI, typename T2>
    static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateLatentEmpty<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateLatentEmpty<T, TI>& out){ }
    template<typename DEVICE, typename T, typename TI, typename T2>
    static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& state, T2 scalar, typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& out){
        out.force[0] += scalar * state.force[0];
        out.force[1] += scalar * state.force[1];
        out.force[2] += scalar * state.force[2];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, T2 scalar, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& out){
        scalar_multiply_accumulate(device, (const LATENT_STATE&)state, scalar, (LATENT_STATE&)out);
        for(int i = 0; i < 3; ++i){
            out.position[i]         += scalar * state.position[i]        ;
            out.orientation[i]      += scalar * state.orientation[i]     ;
            out.linear_velocity[i]  += scalar * state.linear_velocity[i] ;
            out.angular_velocity[i] += scalar * state.angular_velocity[i];
        }
        out.orientation[3] += scalar * state.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename T2, typename LATENT_STATE>
    static void scalar_multiply_accumulate(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, T2 scalar){
        scalar_multiply_accumulate(device, static_cast<const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(state), scalar, static_cast<typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(state));
        for(int i = 0; i < 4; ++i){
            state.rpm[i] += scalar * state.rpm[i];
        }
    }
    template<typename DEVICE, typename T, typename TI>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateLatentEmpty<T, TI>& s1, const typename rl::environments::multirotor::StateLatentEmpty<T, TI>& s2, typename rl::environments::multirotor::StateLatentEmpty<T, TI>& out){ }
    template<typename DEVICE, typename T, typename TI>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& s1, const typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& s2, typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& out){
        out.force[0] = s1.force[0] + s2.force[0];
        out.force[1] = s1.force[1] + s2.force[1];
        out.force[2] = s1.force[2] + s2.force[2];
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& s1, const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& s2, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& out){
        add_accumulate(device, (const LATENT_STATE&)s1, (const LATENT_STATE&)s2, (LATENT_STATE&)out);
        for(int i = 0; i < 3; ++i){
            out.position[i]         = s1.position[i] + s2.position[i];
            out.orientation[i]      = s1.orientation[i] + s2.orientation[i];
            out.linear_velocity[i]  = s1.linear_velocity[i] + s2.linear_velocity[i];
            out.angular_velocity[i] = s1.angular_velocity[i] + s2.angular_velocity[i];
        }
        out.orientation[3] = s1.orientation[3] + s2.orientation[3];
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& s, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& out){
        add_accumulate(device, s, out, out);
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& s1, const typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& s2, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& out){
        add_accumulate(device, static_cast<const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(s1), static_cast<const typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(s2), static_cast<typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&>(out));
        for(int i = 0; i < 4; ++i){
            out.rpm[i] = s1.rpm[i] + s2.rpm[i];
        }
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE>
    static void add_accumulate(DEVICE& device, const typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& s, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& out){
        add_accumulate(device, static_cast<const typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>&>(s), static_cast<typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>&>(out), static_cast<typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>&>(out));
    }
}

#include <backprop_tools/utils/generic/integrators.h>


namespace backprop_tools::rl::environments::multirotor {
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS, typename LATENT_STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics( DEVICE& device, const PARAMETERS& params, const StateLatentEmpty<T, TI>& state, const T* action, StateBase<T, TI, LATENT_STATE>& state_change ) { }
    template<typename DEVICE, typename T, typename TI, typename PARAMETERS, typename LATENT_STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics( DEVICE& device, const PARAMETERS& params, const StateLatentRandomForce<T, TI>& state, const T* action, StateBase<T, TI, LATENT_STATE>& state_change ){
        state_change.linear_velocity[0] += state.force[0] / params.dynamics.mass;
        state_change.linear_velocity[1] += state.force[1] / params.dynamics.mass;
        state_change.linear_velocity[2] += state.force[2] / params.dynamics.mass;

        state_change.force[0] = 0;
        state_change.force[1] = 0;
        state_change.force[2] = 0;
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE, typename PARAMETERS>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(
            DEVICE& device,
            const PARAMETERS& params,
            // state
            const StateBase<T, TI, LATENT_STATE>& state,
            // action
            const T* action,
            // state change
            StateBase<T, TI, LATENT_STATE>& state_change
    ) {
        using STATE = StateBase<T, TI, LATENT_STATE>;

        T thrust[3];
        T torque[3];
        thrust[0] = 0;
        thrust[1] = 0;
        thrust[2] = 0;
        torque[0] = 0;
        torque[1] = 0;
        torque[2] = 0;
        // flops: N*23 => 4 * 23 = 92
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            // flops: 3 + 1 + 3 + 3 + 3 + 4 + 6 = 23
            T rpm = action[i_rotor];
            T thrust_magnitude = params.dynamics.thrust_constants[0] + params.dynamics.thrust_constants[1] * rpm + params.dynamics.thrust_constants[2] * rpm * rpm;
            T rotor_thrust[3];
            utils::vector_operations::scalar_multiply<DEVICE, T, 3>(params.dynamics.rotor_thrust_directions[i_rotor], thrust_magnitude, rotor_thrust);
            utils::vector_operations::add_accumulate<DEVICE, T, 3>(rotor_thrust, thrust);

            utils::vector_operations::scalar_multiply_accumulate<DEVICE, T, 3>(params.dynamics.rotor_torque_directions[i_rotor], thrust_magnitude * params.dynamics.torque_constant, torque);
            utils::vector_operations::cross_product_accumulate<DEVICE, T>(params.dynamics.rotor_positions[i_rotor], rotor_thrust, torque);
        }

        // linear_velocity_global
        state_change.position[0] = state.linear_velocity[0];
        state_change.position[1] = state.linear_velocity[1];
        state_change.position[2] = state.linear_velocity[2];

        // angular_velocity_global
        // flops: 16
        quaternion_derivative<DEVICE, T>(state.orientation, state.angular_velocity, state_change.orientation);

        // linear_acceleration_global
        // flops: 21
        rotate_vector_by_quaternion<DEVICE, T>(state.orientation, thrust, state_change.linear_velocity);
        // flops: 4
        utils::vector_operations::scalar_multiply<DEVICE, T, 3>(state_change.linear_velocity, 1 / params.dynamics.mass);
        utils::vector_operations::add_accumulate<DEVICE, T, 3>(params.dynamics.gravity, state_change.linear_velocity);

        T vector[3];
        T vector2[3];

        // angular_acceleration_local
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J, state.angular_velocity, vector);
        // flops: 6
        utils::vector_operations::cross_product<DEVICE, T>(state.angular_velocity, vector, vector2);
        utils::vector_operations::sub<DEVICE, T, 3>(torque, vector2, vector);
        // flops: 9
        utils::vector_operations::matrix_vector_product<DEVICE, T, 3, 3>(params.dynamics.J_inv, vector, state_change.angular_velocity);
        // total flops: (quadrotor): 92 + 16 + 21 + 4 + 9 + 6 + 9 = 157
//        multirotor_dynamics<DEVICE, T, TI, PARAMETERS>(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
        multirotor_dynamics(device, params, (const typename STATE::LATENT_STATE&)state, action, state_change);
    }
    template<typename DEVICE, typename T, typename TI, typename LATENT_STATE, typename PARAMETERS>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics(
            DEVICE& device,
            const PARAMETERS& params,
            // state
            const StateBaseRotors<T, TI, LATENT_STATE>& state,
            // action
            const T* action,
            // state change
            StateBaseRotors<T, TI, LATENT_STATE>& state_change
    ) {
        for(typename DEVICE::index_t i_rotor = 0; i_rotor < 4; i_rotor++){
            state_change.rpm[i_rotor] = (action[i_rotor] - state.rpm[i_rotor]) * 1/params.dynamics.rpm_time_constant;
        }
        multirotor_dynamics(device, params, (const StateBase<T, TI, LATENT_STATE>&)state, state.rpm, (StateBase<T, TI, LATENT_STATE>&)state_change);

    }
    template<typename DEVICE, typename T, typename PARAMETERS, typename STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void multirotor_dynamics_dispatch(DEVICE& device, const PARAMETERS& params, const STATE& state, const T* action, STATE& state_change) {
        // this dispatch function is required to pass the multirotor dynamics function to the integrator (euler, rk4) as a template parameter (so that it can be inlined/optimized at compile time)
        // If we would try to pass the multirotor_dynamics function directly the state type-based overloading would make the inference of the auto template parameter for the dynamics function in the integrator function impossible
//        multirotor_dynamics<DEVICE, T, typename DEVICE::index_t, typename STATE::LATENT_STATE, PARAMETERS>(device, params, state, action, state_change);
        multirotor_dynamics(device, params, state, action, state_change);
    }

}


namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE, rl::environments::Multirotor<SPEC>){

    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void initial_parameters(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state){
//        T J_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)2, rng);
//        env.current_dynamics.J[0][0] *= J_factor;
//        env.current_dynamics.J[1][1] *= J_factor;
//        env.current_dynamics.J[2][2] *= J_factor;
//        env.current_dynamics.J_inv[0][0] /= J_factor;
//        env.current_dynamics.J_inv[1][1] /= J_factor;
//        env.current_dynamics.J_inv[2][2] /= J_factor;
//        T mass_factor = random::uniform_real_distribution(random_dev, (T)0.5, (T)1.5, rng);
//        env.current_dynamics.mass *= mass_factor;
//        printf("initial state: %f %f %f %f %f %f %f %f %f %f %f %f %f\n", state.state[0], state.state[1], state.state[2], state.state[3], state.state[4], state.state[5], state.state[6], state.state[7], state.state[8], state.state[9], state.state[10], state.state[11], state.state[12]);
        env.current_dynamics = env.parameters.dynamics;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateLatentEmpty<T, TI>& state){ }
    template<typename DEVICE, typename T, typename TI, typename SPEC>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateLatentRandomForce<T, TI>& state){
        state.force[0] = 0;
        state.force[1] = 0;
        state.force[2] = 0;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state){
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        initial_state(device, env, (typename STATE::LATENT_STATE&)state);
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.position[i] = 0;
        }
        state.orientation[0] = 1;
        for(typename DEVICE::index_t i = 1; i < 4; i++){
            state.orientation[i] = 0;
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.linear_velocity[i] = 0;
        }
        for(typename DEVICE::index_t i = 0; i < 3; i++){
            state.angular_velocity[i] = 0;
        }
        initial_parameters(device, env, state);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state){
        initial_state(device, env, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state);
        for(typename DEVICE::index_t i = 0; i < 4; i++){
            state.rpm[i] = 0;
        }
    }
    template<typename DEVICE, typename T, typename TI_H, TI_H HISTORY_LENGTH, typename SPEC, typename LATENT_STATE>
    static void initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBaseRotorsHistory<T, TI_H, HISTORY_LENGTH, LATENT_STATE>& state){
        using TI = typename DEVICE::index_t;
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        initial_state(device, env, (rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>&)state);
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = -1;
            }
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateLatentEmpty<T_S, TI_S>& state, RNG& rng){ }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateLatentRandomForce<T_S, TI_S>& state, RNG& rng){
        typename DEVICE::SPEC::RANDOM random_dev;
        using T = typename SPEC::T;
        auto distribution = env.parameters.disturbances.random_force;
        state.force[0] = random::normal_distribution(random_dev, (T)distribution.mean, (T)distribution.std, rng);
        state.force[1] = random::normal_distribution(random_dev, (T)distribution.mean, (T)distribution.std, rng);
        state.force[2] = random::normal_distribution(random_dev, (T)distribution.mean, (T)distribution.std, rng);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename RNG, typename LATENT_STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, RNG& rng){
        typename DEVICE::SPEC::MATH math_dev;
        typename DEVICE::SPEC::RANDOM random_dev;
        using STATE = typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>;
        sample_initial_state(device, env, (typename STATE::LATENT_STATE&)state, rng);
        bool guidance = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng) < env.parameters.mdp.init.guidance;
        if(!guidance){
            for(TI i = 0; i < 3; i++){
                state.position[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_position, env.parameters.mdp.init.max_position, rng);
            }
        }
        else{
            for(TI i = 0; i < 3; i++){
                state.position[i] = 0;
            }
        }
        // https://web.archive.org/web/20181126051029/http://planning.cs.uiuc.edu/node198.html
        if(env.parameters.mdp.init.max_angle > 0 && !guidance){
            do{
                T u[3];
                for(TI i = 0; i < 3; i++){
                    u[i] = random::uniform_real_distribution(random_dev, (T)0, (T)1, rng);
                }
                state.orientation[0] = math::sqrt(math_dev, 1-u[0]) * math::sin(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[1] = math::sqrt(math_dev, 1-u[0]) * math::cos(math_dev, 2*math::PI<T>*u[1]);
                state.orientation[2] = math::sqrt(math_dev,   u[0]) * math::sin(math_dev, 2*math::PI<T>*u[2]);
                state.orientation[3] = math::sqrt(math_dev,   u[0]) * math::cos(math_dev, 2*math::PI<T>*u[2]);
            } while(math::abs(math_dev, 2*math::acos(math_dev, state.orientation[0])) > env.parameters.mdp.init.max_angle);
        }
        else{
            state.orientation[0] = 1;
            state.orientation[1] = 0;
            state.orientation[2] = 0;
            state.orientation[3] = 0;
        }
        for(TI i = 0; i < 3; i++){
            state.linear_velocity[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_linear_velocity, env.parameters.mdp.init.max_linear_velocity, rng);
        }
        for(TI i = 0; i < 3; i++){
            state.angular_velocity[i] = random::uniform_real_distribution(random_dev, -env.parameters.mdp.init.max_angular_velocity, env.parameters.mdp.init.max_angular_velocity, rng);
        }
        initial_parameters(device, env, state);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, RNG& rng){
        sample_initial_state(device, env, (typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state, rng);
        T min_rpm, max_rpm;
        if(env.parameters.mdp.init.relative_rpm){
            min_rpm = (env.parameters.mdp.init.min_rpm + 1)/2 * (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) + env.parameters.dynamics.action_limit.min;
            max_rpm = (env.parameters.mdp.init.max_rpm + 1)/2 * (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) + env.parameters.dynamics.action_limit.min;
        }
        else{
            min_rpm = env.parameters.mdp.init.min_rpm < 0 ? env.parameters.dynamics.action_limit.min : env.parameters.mdp.init.min_rpm;
            max_rpm = env.parameters.mdp.init.max_rpm < 0 ? env.parameters.dynamics.action_limit.max : env.parameters.mdp.init.max_rpm;
            if(max_rpm > env.parameters.dynamics.action_limit.max){
                max_rpm = env.parameters.dynamics.action_limit.max;
            }
            if(min_rpm > max_rpm){
                min_rpm = max_rpm;
            }
        }
        for(TI i = 0; i < 4; i++){
            state.rpm[i] = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), min_rpm, max_rpm, rng);
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename LATENT_STATE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, rl::environments::Multirotor<SPEC>& env, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, RNG& rng){
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        sample_initial_state(device, env, (typename rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>&)state, rng);
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                state.action_history[step_i][action_i] = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min) / (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
            }
        }
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng, bool privileged = false){
        using ENVIRONMENT = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::ROWS == 1);
        using STATE = typename ENVIRONMENT::State;
//        constexpr T position_noise_std = 0.003;
//        constexpr T orientation_noise_std = 0.100;
//        constexpr T linear_velocity_noise = 0.01;
//        constexpr T angular_velocity_noise = 0.003;
//        add_scalar(device, device.logger, "quaternion_w", state.state[3], 1000);

        // Position
        TI current_observation_i = 0;
        for(TI i = 0; i < 3; i++){
            T noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.position, rng) : 0;
            set(observation, 0, current_observation_i, state.position[i] + noise);
            current_observation_i += 1;
        }

        // Orientation
        if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::Normal){
            static_assert(OBS_SPEC::COLS == 13);
            for(TI i = 0; i < 4; i++){
                T noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                set(observation, 0, current_observation_i, state.orientation[i] + noise);
                current_observation_i += 1;
            }
            if constexpr(SPEC::STATIC_PARAMETERS::ENFORCE_POSITIVE_QUATERNION){
                if(get(observation, 0, current_observation_i - 4) < 0){
                    for(TI i = 0; i < 4; i++){
                        TI quat_observation_i = current_observation_i - 4 + i;
                        set(observation, 0, quat_observation_i, -get(observation, 0, quat_observation_i));
                    }
                }
            }
            else{
                if constexpr(SPEC::STATIC_PARAMETERS::RANDOMIZE_QUATERNION_SIGN){
                    if(random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), 0, 1, rng) == 0){
                        for(TI i = 0; i < 4; i++){
                            TI quat_observation_i = current_observation_i - 4 + i;
                            set(observation, 0, quat_observation_i, -get(observation, 0, quat_observation_i));
                        }
                    }
                }
            }
        }
        else{
            if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::DoubleQuaternion){
                static_assert(OBS_SPEC::COLS == 17);
                typename SPEC::T sign = state.orientation[0] > 0 ? 1 : -1;
                for(TI i = 0; i < 4; i++){
                    set(observation, 0, current_observation_i+0+i,   sign * state.orientation[i]);
                    set(observation, 0, current_observation_i+4+i, - sign * state.orientation[i]);
                }
                current_observation_i += 8;
            }
            else{
                if constexpr(SPEC::STATIC_PARAMETERS::OBSERVATION_TYPE == rl::environments::multirotor::ObservationType::RotationMatrix){
                    static_assert(OBS_SPEC::COLS == 18);
                    const typename SPEC::T* q = state.orientation;
                    T noise;
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 0, (1 - 2*q[2]*q[2] - 2*q[3]*q[3]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 1, (    2*q[1]*q[2] - 2*q[0]*q[3]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 2, (    2*q[1]*q[3] + 2*q[0]*q[2]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 3, (    2*q[1]*q[2] + 2*q[0]*q[3]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 4, (1 - 2*q[1]*q[1] - 2*q[3]*q[3]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 5, (    2*q[2]*q[3] - 2*q[0]*q[1]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 6, (    2*q[1]*q[3] - 2*q[0]*q[2]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 7, (    2*q[2]*q[3] + 2*q[0]*q[1]) + noise);
                    noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.orientation, rng) : 0;
                    set(observation, 0, current_observation_i + 8, (1 - 2*q[1]*q[1] - 2*q[2]*q[2]) + noise);
                    current_observation_i += 9;
                }
            }
        }

        // Linear Velocity
        for(TI i = 0; i < 3; i++){
            T noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.linear_velocity, rng) : 0;
            set(observation, 0, current_observation_i, state.linear_velocity[i] + noise);
            current_observation_i += 1;
        }

        // Angular Velocity
        for(TI i = 0; i < 3; i++){
            T noise = !privileged ? random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.observation_noise.angular_velocity, rng) : 0;
            set(observation, 0, current_observation_i, state.angular_velocity[i] + noise);
            current_observation_i += 1;
        }
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        observe(device, env, state, observation, rng, true);
    }
    template<typename DEVICE, typename T_H, typename TI_H, TI_H HISTORY_LENGTH, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBaseRotorsHistory<T_H, TI_H, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == MULTIROTOR::OBSERVATION_DIM);
        using TI = typename DEVICE::index_t;
        auto base_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION>{}, 0, 0);
        observe_privileged(device, env, (rl::environments::multirotor::StateBaseRotors<T_H, TI_H, LATENT_STATE>&)state, base_observation, rng);
        auto action_history_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_ACTION_HISTORY>{}, 0, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION);
        for(TI step_i = 0; step_i < HISTORY_LENGTH; step_i++){
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                set(action_history_observation, 0, step_i*MULTIROTOR::ACTION_DIM + action_i, state.action_history[step_i][action_i]);
            }
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename SPEC, typename OBS_SPEC, typename LATENT_STATE, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == MULTIROTOR::OBSERVATION_DIM_PRIVILEGED);
        auto base_observation = view(device, observation, matrix::ViewSpec<1, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION>{}, 0, 0);
        observe(device, env, state, base_observation, rng);
        for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
            T action_value = (state.rpm[action_i] - env.parameters.dynamics.action_limit.min)/(env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) * 2 - 1;
            set(observation, 0, MULTIROTOR::OBSERVATION_DIM_BASE + MULTIROTOR::OBSERVATION_DIM_ORIENTATION + action_i, action_value);
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, typename LATENT_STATE, typename SPEC, typename ACTION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void step_action_history(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateBase<T_S, TI_S, LATENT_STATE>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateBase<T_S, TI_S, LATENT_STATE>& next_state) { }
    template<typename DEVICE, typename T_S, typename TI_S, typename LATENT_STATE, TI_S HISTORY_LENGTH, typename SPEC, typename ACTION_SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void step_action_history(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& next_state) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        using TI = typename DEVICE::index_t;
        static_assert(ACTION_SPEC::COLS == MULTIROTOR::ACTION_DIM);
        if constexpr(HISTORY_LENGTH > 0){
            for(TI step_i = 0; step_i < HISTORY_LENGTH-1; step_i++){
                for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                    next_state.action_history[step_i][action_i] = state.action_history[step_i+1][action_i];
                }
            }
            for(TI action_i = 0; action_i < MULTIROTOR::ACTION_DIM; action_i++){
                next_state.action_history[HISTORY_LENGTH-1][action_i] = get(action, 0, action_i);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename T, typename TI, typename LATENT_STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void normalize(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state) {
        T quaternion_norm = 0;
        for(TI state_i = 0; state_i < 4; state_i++){
            quaternion_norm += state.orientation[state_i] * state.orientation[state_i];
        }
        quaternion_norm = math::sqrt(typename DEVICE::SPEC::MATH(), quaternion_norm);
        for(TI state_i = 0; state_i < 4; state_i++){
            state.orientation[state_i] /= quaternion_norm;
        }
    }
    template<typename DEVICE, typename SPEC, typename T, typename TI, typename LATENT_STATE>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void normalize(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state) {
        using MULTIROTOR = rl::environments::Multirotor<SPEC>;
        normalize(device, env, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state);
        for(typename DEVICE::index_t rpm_i = 0; rpm_i < MULTIROTOR::ACTION_DIM; rpm_i++){
            state.rpm[rpm_i] = math::clamp(typename DEVICE::SPEC::MATH{}, state.rpm[rpm_i], env.parameters.dynamics.action_limit.min, env.parameters.dynamics.action_limit.max);
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void step_latent(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateLatentEmpty<T_S, TI_S>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateLatentEmpty<T_S, TI_S>& next_state, RNG& rng) { }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename T_S, typename TI_S, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void step_latent(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::multirotor::StateLatentRandomForce<T_S, TI_S>& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::multirotor::StateLatentRandomForce<T_S, TI_S>& next_state, RNG& rng) {
        next_state.force[0] = state.force[0];
        next_state.force[1] = state.force[1];
        next_state.force[2] = state.force[2];
    }
    // todo: make state const again
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T step(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Multirotor<SPEC>::State& next_state, RNG& rng) {
        using STATE = typename rl::environments::Multirotor<SPEC>::State;
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr auto STATE_DIM = STATE::DIM;
        constexpr auto ACTION_DIM = rl::environments::Multirotor<SPEC>::ACTION_DIM;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ACTION_DIM);
        T action_scaled[ACTION_DIM];

        for(TI action_i = 0; action_i < ACTION_DIM; action_i++){
            T half_range = (env.parameters.dynamics.action_limit.max - env.parameters.dynamics.action_limit.min) / 2;
            T action_noisy = get(action, 0, action_i);
            action_noisy += random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, env.parameters.mdp.action_noise.normalized_rpm, rng);
            action_noisy = math::clamp(typename DEVICE::SPEC::MATH(), action_noisy, -(T)1, (T)1);
            action_scaled[action_i] = action_noisy * half_range + env.parameters.dynamics.action_limit.min + half_range;
//            state.rpm[action_i] = action_scaled[action_i];
        }
        utils::integrators::rk4<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE>>(device, env.parameters, state, action_scaled, env.parameters.integration.dt, next_state);
//        utils::integrators::euler<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM, rl::environments::multirotor::multirotor_dynamics_dispatch<DEVICE, typename SPEC::T, typename SPEC::PARAMETERS, STATE, ACTION_DIM>>(device, env.parameters, state, action_scaled, env.parameters.integration.dt, next_state);

        step_latent(device, env, state, action, next_state, rng);

        normalize(device, env, next_state);

        step_action_history(device, env, state, action, next_state);

        return env.parameters.integration.dt;
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Multirotor<SPEC>& env, const typename rl::environments::Multirotor<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        if(env.parameters.mdp.termination.enabled){
            for(TI i = 0; i < 3; i++){
                if(
                    math::abs(typename DEVICE::SPEC::MATH(), state.position[i]) > env.parameters.mdp.termination.position_threshold ||
                    math::abs(typename DEVICE::SPEC::MATH(), state.linear_velocity[i]) > env.parameters.mdp.termination.linear_velocity_threshold ||
                    math::abs(typename DEVICE::SPEC::MATH(), state.angular_velocity[i]) > env.parameters.mdp.termination.angular_velocity_threshold
                ){
                    return true;
                }
            }
        }
        return false;
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void serialize(DEVICE& device, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        set(flat_state, 0, 0, state.position[0]);
        set(flat_state, 0, 1, state.position[1]);
        set(flat_state, 0, 2, state.position[2]);
        set(flat_state, 0, 3, state.orientation[0]);
        set(flat_state, 0, 4, state.orientation[1]);
        set(flat_state, 0, 5, state.orientation[2]);
        set(flat_state, 0, 6, state.orientation[3]);
        set(flat_state, 0, 7, state.linear_velocity[0]);
        set(flat_state, 0, 8, state.linear_velocity[1]);
        set(flat_state, 0, 9, state.linear_velocity[2]);
        set(flat_state, 0, 10, state.angular_velocity[0]);
        set(flat_state, 0, 11, state.angular_velocity[1]);
        set(flat_state, 0, 12, state.angular_velocity[2]);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        state.position[0] = get(flat_state, 0, 0);
        state.position[1] = get(flat_state, 0, 1);
        state.position[2] = get(flat_state, 0, 2);
        state.orientation[0] = get(flat_state, 0, 3);
        state.orientation[1] = get(flat_state, 0, 4);
        state.orientation[2] = get(flat_state, 0, 5);
        state.orientation[3] = get(flat_state, 0, 6);
        state.linear_velocity[0] = get(flat_state, 0, 7);
        state.linear_velocity[1] = get(flat_state, 0, 8);
        state.linear_velocity[2] = get(flat_state, 0, 9);
        state.angular_velocity[0] = get(flat_state, 0, 10);
        state.angular_velocity[1] = get(flat_state, 0, 11);
        state.angular_velocity[2] = get(flat_state, 0, 12);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void serialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, 13>{}, 0, 0);
        serialize(device, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state, state_base_flat);
        constexpr TI offset = rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>::DIM;
        set(flat_state, 0, offset + 0, state.rpm[0]);
        set(flat_state, 0, offset + 1, state.rpm[1]);
        set(flat_state, 0, offset + 2, state.rpm[2]);
        set(flat_state, 0, offset + 3, state.rpm[3]);
    }
    template<typename DEVICE, typename T, typename TI, typename SPEC, typename LATENT_STATE>
    static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBaseRotors<T, TI, LATENT_STATE>;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        constexpr TI OFFSET = rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>::DIM;
        auto state_base_rotors_flat = view(device, flat_state, matrix::ViewSpec<1, OFFSET>{}, 0, 0);
        deserialize(device, (rl::environments::multirotor::StateBase<T, TI, LATENT_STATE>&)state, state_base_rotors_flat);
        state.rpm[0] = get(flat_state, 0, OFFSET + 0);
        state.rpm[1] = get(flat_state, 0, OFFSET + 1);
        state.rpm[2] = get(flat_state, 0, OFFSET + 2);
        state.rpm[3] = get(flat_state, 0, OFFSET + 3);
    }
    template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename SPEC, typename LATENT_STATE>
    static void serialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>;
        using TI = typename DEVICE::index_t;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        constexpr TI OFFSET = rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>::DIM;
        auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, OFFSET>{}, 0, 0);
        serialize(device, (rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>&)state, state_base_flat);
        for(TI step_i = 0; step_i < HISTORY_LENGTH; ++step_i){
            set(flat_state, 0, OFFSET + step_i * 4 + 0, state.action_history[step_i][0]);
            set(flat_state, 0, OFFSET + step_i * 4 + 1, state.action_history[step_i][1]);
            set(flat_state, 0, OFFSET + step_i * 4 + 2, state.action_history[step_i][2]);
            set(flat_state, 0, OFFSET + step_i * 4 + 3, state.action_history[step_i][3]);
        }
    }
    template<typename DEVICE, typename T_S, typename TI_S, TI_S HISTORY_LENGTH, typename SPEC, typename LATENT_STATE>
    static void deserialize(DEVICE& device, typename rl::environments::multirotor::StateBaseRotorsHistory<T_S, TI_S, HISTORY_LENGTH, LATENT_STATE>& state, Matrix<SPEC>& flat_state){
        using STATE = typename rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>;
        using TI = typename DEVICE::index_t;
        static_assert(SPEC::ROWS == 1);
        static_assert(SPEC::COLS == STATE::DIM);
        auto state_base_flat = view(device, flat_state, matrix::ViewSpec<1, 13>{}, 0, 0);
        deserialize(device, (rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>&)state, state_base_flat);
        constexpr TI offset = rl::environments::multirotor::StateBaseRotors<T_S, TI_S, LATENT_STATE>::DIM;
        for(TI step_i = 0; step_i < HISTORY_LENGTH; ++step_i){
            state.rpm_history[step_i][0] = get(flat_state, 0, offset + step_i * 4 + 0);
            state.rpm_history[step_i][1] = get(flat_state, 0, offset + step_i * 4 + 1);
            state.rpm_history[step_i][2] = get(flat_state, 0, offset + step_i * 4 + 2);
            state.rpm_history[step_i][3] = get(flat_state, 0, offset + step_i * 4 + 3);
        }
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
