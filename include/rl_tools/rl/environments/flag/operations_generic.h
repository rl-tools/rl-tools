#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_FLAG_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_FLAG_OPERATIONS_GENERIC_H
#include "environment.h"
#include "../operations_generic.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void malloc(DEVICE& device, const rl::environments::Flag<SPEC>& env){ }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void free(DEVICE& device, const rl::environments::Flag<SPEC>& env){ }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void init(DEVICE& device, const rl::environments::Flag<SPEC>& env){ }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_parameters(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, RNG& rng){
        using T = typename SPEC::T;
        parameters.flag_position[0] = random::uniform_real_distribution(device.random, (T)0, SPEC::PARAMETERS::BOARD_SIZE, rng);
        parameters.flag_position[1] = random::uniform_real_distribution(device.random, (T)0, SPEC::PARAMETERS::BOARD_SIZE, rng);
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_parameters(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters){
        parameters.flag_position[0] = SPEC::PARAMETERS::BOARD_SIZE / 2;
        parameters.flag_position[1] = SPEC::PARAMETERS::BOARD_SIZE / 2;
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_state(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, typename rl::environments::Flag<SPEC>::State& state){
        state.position[0] = SPEC::PARAMETERS::BOARD_SIZE / 2;
        state.position[1] = SPEC::PARAMETERS::BOARD_SIZE / 2;
        state.velocity[0] = 0;
        state.velocity[1] = 0;
        state.state_machine = rl::environments::Flag<SPEC>::State::StateMachine::INITIAL;
    }
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT static void initial_state(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, typename rl::environments::FlagMemory<SPEC>::State& state){
        initial_state(device, static_cast<const rl::environments::Flag<SPEC>&>(env), parameters, state);
        state.first_step = true;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, typename rl::environments::Flag<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        state.position[0] = random::uniform_real_distribution(device.random, (T)0, SPEC::PARAMETERS::BOARD_SIZE, rng);
        state.position[1] = random::uniform_real_distribution(device.random, (T)0, SPEC::PARAMETERS::BOARD_SIZE, rng);
        state.velocity[0] = 0;
        state.velocity[1] = 0;
        state.state_machine = rl::environments::Flag<SPEC>::State::StateMachine::INITIAL;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, typename rl::environments::FlagMemory<SPEC>::State& state, RNG& rng){
        sample_initial_state(device, static_cast<const rl::environments::Flag<SPEC>&>(env), parameters, state, rng);
        state.first_step = true;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, const typename rl::environments::Flag<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Flag<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 2);
        using namespace rl::environments::pendulum;
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;
        using STATE = typename rl::environments::Flag<SPEC>::State;
        T u_normalised[2];
        u_normalised[0] = math::clamp(device.math, get(action, 0, 0), (T)-1, (T)1);
        u_normalised[1] = math::clamp(device.math, get(action, 0, 1), (T)-1, (T)1);
        T u[2];
        u[0] = PARAMS::MAX_ACCELERATION * u_normalised[0];
        u[1] = PARAMS::MAX_ACCELERATION * u_normalised[1];
        next_state.position[0] = state.position[0] + state.velocity[0] * PARAMS::DT;
        next_state.position[1] = state.position[1] + state.velocity[1] * PARAMS::DT;
        next_state.velocity[0] = state.velocity[0] + u[0] * PARAMS::DT;
        next_state.velocity[1] = state.velocity[1] + u[1] * PARAMS::DT;
        next_state.position[0] = math::clamp(device.math, next_state.position[0], (T)0, PARAMS::BOARD_SIZE);
        next_state.position[1] = math::clamp(device.math, next_state.position[1], (T)0, PARAMS::BOARD_SIZE);
        next_state.velocity[0] = math::clamp(device.math, next_state.velocity[0], (T)-PARAMS::MAX_VELOCITY, PARAMS::MAX_VELOCITY);
        next_state.velocity[1] = math::clamp(device.math, next_state.velocity[1], (T)-PARAMS::MAX_VELOCITY, PARAMS::MAX_VELOCITY);

        T distance_to_flag = math::sqrt(device.math, (next_state.position[0] - parameters.flag_position[0]) * (next_state.position[0] - parameters.flag_position[0]) + (next_state.position[1] - parameters.flag_position[1]) * (next_state.position[1] - parameters.flag_position[1]));
        T distance_to_origin = math::sqrt(device.math, next_state.position[0] * next_state.position[0] + next_state.position[1] * next_state.position[1]);
        switch(state.state_machine){
            case STATE::StateMachine::INITIAL:
                if(distance_to_flag < PARAMS::FLAG_DISTANCE_THRESHOLD){
                    next_state.state_machine = STATE::StateMachine::FLAG_VISITED;
                }
                else{
                    next_state.state_machine = STATE::StateMachine::INITIAL;
                }
                break;
            case STATE::StateMachine::FLAG_VISITED:
                if(distance_to_origin < PARAMS::FLAG_DISTANCE_THRESHOLD){
                    next_state.state_machine = STATE::StateMachine::ORIGIN_VISITED;
                }
                else{
                    next_state.state_machine = STATE::StateMachine::FLAG_VISITED;
                }
                break;
            case STATE::StateMachine::ORIGIN_VISITED:
                if(distance_to_flag < PARAMS::FLAG_DISTANCE_THRESHOLD){
                    next_state.state_machine = STATE::StateMachine::FLAG_VISITED_AGAIN;
                }
                else{
                    next_state.state_machine = STATE::StateMachine::ORIGIN_VISITED;
                }
                break;
            default:
                std::cout << "unexpected state: " << static_cast<typename DEVICE::index_t>(state.state_machine) << std::endl;
                std::exit(1);
                break;
        }
        return SPEC::PARAMETERS::DT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, const typename rl::environments::FlagMemory<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::FlagMemory<SPEC>::State& next_state, RNG& rng){
        typename SPEC::T dt = step(device, static_cast<const rl::environments::Flag<SPEC>&>(env), parameters, state, action, next_state, rng);
        next_state.first_step = false;
        return dt;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, const typename rl::environments::Flag<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Flag<SPEC>::State& next_state, RNG& rng){
        using namespace rl::environments::pendulum;
        using T = typename SPEC::T;
        using ENVIRONMENT = rl::environments::Flag<SPEC>;
        using STATE = typename rl::environments::Flag<SPEC>::State;
        T distance_to_flag = math::sqrt(device.math, (next_state.position[0] - parameters.flag_position[0]) * (next_state.position[0] - parameters.flag_position[0]) + (next_state.position[1] - parameters.flag_position[1]) * (next_state.position[1] - parameters.flag_position[1]));
        T distance_to_origin = math::sqrt(device.math, next_state.position[0] * next_state.position[0] + next_state.position[1] * next_state.position[1]);
        T reward = 0;
        if(next_state.state_machine == STATE::StateMachine::FLAG_VISITED_AGAIN){
            reward = 1000;
        }
        else{
            reward = -1;
        }
        return reward/ENVIRONMENT::EPISODE_STEP_LIMIT;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, const typename rl::environments::FlagMemory<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::FlagMemory<SPEC>::State& next_state, RNG& rng){
        using T = typename SPEC::T;
        using ENVIRONMENT = rl::environments::Flag<SPEC>;
        using STATE = typename rl::environments::Flag<SPEC>::State;
//        T reward = 0;
//        if(next_state.state_machine == STATE::StateMachine::FLAG_VISITED){
//            reward = 1000;
//        }
//        else{
//            reward = -1;
//        }
//        return reward/ENVIRONMENT::EPISODE_STEP_LIMIT;
        //
        // return -math::sqrt(device.math, (next_state.position[0] - parameters.flag_position[0]) * (next_state.position[0] - parameters.flag_position[0]) + (next_state.position[1] - parameters.flag_position[1]) * (next_state.position[1] - parameters.flag_position[1]));
        T x_diff = get(action, 0, 0) - parameters.flag_position[0] / SPEC::PARAMETERS::BOARD_SIZE;
        T y_diff = get(action, 0, 1) - parameters.flag_position[1] / SPEC::PARAMETERS::BOARD_SIZE;
        T distance = math::sqrt(device.math, x_diff * x_diff + y_diff * y_diff);
        T reward = -distance;
        return reward;
    }

    template<typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Flag<SPEC>& env, const typename rl::environments::Flag<SPEC>::Parameters& parameters, const typename rl::environments::Flag<SPEC>::State& state, const typename rl::environments::flag::Observation<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 7);
        using T = typename SPEC::T;
        using STATE_MACHINE = typename rl::environments::Flag<SPEC>::State::StateMachine;
        set(observation, 0, 0, state.position[0]);
        set(observation, 0, 1, state.position[1]);
        set(observation, 0, 2, state.velocity[0]);
        set(observation, 0, 3, state.velocity[1]);
        switch(state.state_machine){
            case STATE_MACHINE::INITIAL:
                set(observation, 0, 4, 1);
                set(observation, 0, 5, -1);
                set(observation, 0, 6, -1);
                break;
            case STATE_MACHINE::FLAG_VISITED:
                set(observation, 0, 4, -1);
                set(observation, 0, 5, 1);
                set(observation, 0, 6, -1);
                break;
            case STATE_MACHINE::ORIGIN_VISITED:
                set(observation, 0, 4, -1);
                set(observation, 0, 5, -1);
                set(observation, 0, 6, 1);
                break;
            case STATE_MACHINE::FLAG_VISITED_AGAIN:
                set(observation, 0, 4, 0);
                set(observation, 0, 5, 0);
                set(observation, 0, 6, 0);
                break;
        }
    }
    template<typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Flag<SPEC>& env, const typename rl::environments::Flag<SPEC>::Parameters& parameters, const typename rl::environments::Flag<SPEC>::State& state, const typename rl::environments::flag::ObservationPrivileged<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 9);
        using T = typename SPEC::T;
        auto observation_view = view(device, observation, matrix::ViewSpec<1, 7>{});
        observe(device, env, parameters, state, typename rl::environments::flag::Observation<OBS_TYPE_SPEC>{}, observation_view, rng);
        set(observation, 0, 7, parameters.flag_position[0]);
        set(observation, 0, 8, parameters.flag_position[1]);
    }
    template<typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, const typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, const typename rl::environments::FlagMemory<SPEC>::State& state, const typename rl::environments::flag::ObservationMemory<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 9);
        using T = typename SPEC::T;
        auto observation_view = view(device, observation, matrix::ViewSpec<1, 7>{});
        observe(device, env, parameters, state, typename rl::environments::flag::Observation<OBS_TYPE_SPEC>{}, observation_view, rng);
        if(state.first_step){
            set(observation, 0, 7, parameters.flag_position[0]);
            set(observation, 0, 8, parameters.flag_position[1]);
        }
        else{
            set(observation, 0, 7, -1);
            set(observation, 0, 8, -1);
        }
//        set(observation, 0, 7, parameters.flag_position[0]);
//        set(observation, 0, 8, parameters.flag_position[1]);
    }
    template<typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, const typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, const typename rl::environments::FlagMemory<SPEC>::State& state, const typename rl::environments::flag::ObservationMemoryPrivileged<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG& rng) {
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 9);
        using T = typename SPEC::T;
        auto observation_view = view(device, observation, matrix::ViewSpec<1, 7>{});
        observe(device, env, parameters, state, typename rl::environments::flag::Observation<OBS_TYPE_SPEC>{}, observation_view, rng);
        set(observation, 0, 7, parameters.flag_position[0]);
        set(observation, 0, 8, parameters.flag_position[1]);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Flag<SPEC>& env, typename rl::environments::Flag<SPEC>::Parameters& parameters, const typename rl::environments::Flag<SPEC>::State state, RNG& rng){
        return state.state_machine == rl::environments::Flag<SPEC>::State::StateMachine::FLAG_VISITED_AGAIN;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::FlagMemory<SPEC>& env, typename rl::environments::FlagMemory<SPEC>::Parameters& parameters, const typename rl::environments::FlagMemory<SPEC>::State state, RNG& rng){
        return state.state_machine == rl::environments::Flag<SPEC>::State::StateMachine::FLAG_VISITED;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
