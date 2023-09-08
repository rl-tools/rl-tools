#include "../../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_GENERIC_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_GENERIC_H
#include "car.h"
#include "../operations_generic.h"
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::environments::car {
    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT T f_mod_python(const DEVICE& dev, T a, T b){
        return a - b * math::floor(dev, a / b);
    }

    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT T angle_normalize(const DEVICE& dev, T x){
        return f_mod_python(dev, (x + math::PI<T>), (2 * math::PI<T>)) - math::PI<T>;
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template<typename DEVICE, typename SPEC>
    static void init(DEVICE& device, const rl::environments::Car<SPEC>& env){ }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Car<SPEC>& env, typename rl::environments::Car<SPEC>::State& state){
        state.x = 0;
        state.y = 0;
        state.mu = 0;
        state.vx = 0;
        state.vy = 0;
        state.omega = 0;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Car<SPEC>& env, typename rl::environments::Car<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        initial_state(device, env, state);
        constexpr T dist = 0.2;
        state.x = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -dist, dist, rng);
        state.y = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -dist, dist, rng);
        state.mu = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -math::PI<T>, math::PI<T>, rng);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State& s, const Matrix<ACTION_SPEC>& action, typename rl::environments::Car<SPEC>::State& next_state, RNG& rng) {
        using ENVIRONMENT = rl::environments::Car<SPEC>;
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == ENVIRONMENT::ACTION_DIM);
        using namespace rl::environments::car;
        using T = typename SPEC::T;

        T throttle_break = get(action, 0, 0);
        T delta = get(action, 0, 1);
        
        auto& p = env.parameters;
        
        T alpha_f = math::atan2(typename DEVICE::SPEC::MATH(), (s.vy + p.lf * s.omega), s.vx) - delta;
        T alpha_r = math::atan2(typename DEVICE::SPEC::MATH(), (s.vy - p.lr * s.omega), s.vx);
        T FnF = p.lr / (p.lf + p.lr) * p.m * p.g;
        T FnR = p.lf / (p.lf + p.lr) * p.m * p.g;
        T FyF = s.vx > p.vt || s.vy > p.vt ? FnF * p.tf.D * sin(p.tf.C * atan(p.tf.B * (-alpha_f))) : 0;
        T FyR = s.vx > p.vt || s.vy > p.vt ? FnR * p.tr.D * sin(p.tr.C * atan(p.tr.B * (-alpha_r))) : 0;

        T Fx = p.cm * throttle_break - p.cr0 - p.cr2 * s.vx * s.vx;
        next_state = s;
        next_state.x     += p.dt * (cos(s.mu) * s.vx - sin(s.mu) * s.vy);
        next_state.y     += p.dt * (sin(s.mu) * s.vx + cos(s.mu) * s.vy);
        next_state.mu    += p.dt * (s.omega);
        next_state.vx    += p.dt * (1/p.m*(Fx - FyF * sin(delta)) + s.omega * s.vy);
        next_state.vy    += p.dt * (1/p.m*(FyF * cos(delta) + FyR) - s.omega * s.vx);
        next_state.omega += p.dt * (1/p.I*(FyF * p.lf * cos(delta) - FyR * p.lr));

        return p.dt;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Car<SPEC>::State& next_state, RNG& rng){
        using namespace rl::environments::car;
        typedef typename SPEC::T T;
        T angle_norm = angle_normalize(typename DEVICE::SPEC::MATH(), state.mu);
        T cost = 0.1*angle_norm * angle_norm + state.x * state.x + state.y * state.y;
        return math::exp(typename DEVICE::SPEC::MATH(), -5*cost);
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::CarTrack<SPEC>& env, const typename rl::environments::Car<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Car<SPEC>::State& next_state, RNG& rng){
        using namespace rl::environments::car;
        typedef typename SPEC::T T;
        return state.vx;
    }

    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::Car<SPEC>;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == ENVIRONMENT::OBSERVATION_DIM);
        typedef typename SPEC::T T;
        set(observation, 0, 0, state.x);
        set(observation, 0, 1, state.y);
        set(observation, 0, 2, state.mu);
        set(observation, 0, 3, state.vx);
        set(observation, 0, 4, state.vy);
        set(observation, 0, 5, state.omega);
    }
    template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::CarTrack<SPEC>& env, const typename rl::environments::CarTrack<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
        using ENVIRONMENT = rl::environments::CarTrack<SPEC>;
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == ENVIRONMENT::OBSERVATION_DIM);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        auto observation_base = view(device, observation, matrix::ViewSpec<1, rl::environments::Car<SPEC>::OBSERVATION_DIM>{});
        observe(device, (rl::environments::Car<SPEC>&)env, (typename rl::environments::Car<SPEC>::State&)state, observation_base, rng);
        constexpr TI N_DIRECTIONS = 3;
        constexpr TI NUM_STEPS = 50;
        constexpr T step_size = SPEC::TRACK_SCALE / 2;
        T directions[N_DIRECTIONS] = {-20/180.0*math::PI<T>, 0, 20/180.0*math::PI<T>};

        for(TI direction_i=0; direction_i < N_DIRECTIONS; direction_i++){
            T direction = directions[direction_i];
            T direction_x = cos(state.mu + direction) * step_size;
            T direction_y = sin(state.mu + direction) * step_size;
            T distance = NUM_STEPS * step_size;

            for(TI step_i = 0; step_i < NUM_STEPS; step_i++){
                T x_coord = (+state.x + direction_x * step_i + SPEC::TRACK_SCALE * SPEC::WIDTH  / 2.0) / ((T)SPEC::TRACK_SCALE);
                T y_coord = (-state.y - direction_y * step_i + SPEC::TRACK_SCALE * SPEC::HEIGHT / 2.0) / ((T)SPEC::TRACK_SCALE);
                if(!env.parameters.track[(TI)y_coord][(TI)x_coord]){
                    distance = step_i * step_size;
                    break;
                }
            }
            set(observation, 0, 6 + direction_i, distance / (NUM_STEPS * step_size));
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State state, RNG& rng){
        using T = typename SPEC::T;
        return state.x > 1.0 || state.x < -1.0 || state.y > 1.0 || state.y < -1.0;
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::CarTrack<SPEC>& env, const typename rl::environments::CarTrack<SPEC>::State state, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        T x_coord = (state.x + SPEC::TRACK_SCALE * SPEC::WIDTH / 2.0) / ((T)SPEC::TRACK_SCALE);
        T y_coord = (-state.y + SPEC::TRACK_SCALE * SPEC::HEIGHT / 2.0) / ((T)SPEC::TRACK_SCALE);
        if(x_coord > 0 && x_coord < SPEC::WIDTH && y_coord > 0 && y_coord < SPEC::HEIGHT){
            return !env.parameters.track[(TI)y_coord][(TI)x_coord];
        }
        else{
            return true;
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::CarTrack<SPEC>& env, typename rl::environments::Car<SPEC>::State& state, RNG& rng){
        using T = typename SPEC::T;
        initial_state(device, env, state);
        do{
            state.x = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::BOUND_X_LOWER, SPEC::BOUND_X_UPPER, rng);
            state.y = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::BOUND_Y_LOWER, SPEC::BOUND_Y_UPPER, rng);
            state.mu = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -math::PI<T>, math::PI<T>, rng);
        } while(terminated(device, env, state, rng));
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
