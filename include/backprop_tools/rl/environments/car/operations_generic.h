#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_GENERIC
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_CAR_OPERATIONS_GENERIC
#include "car.h"
#include "../operations_generic.h"
namespace backprop_tools::rl::environments::car {
    template <typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT T clip(T x, T min, T max){
        x = x < min ? min : (x > max ? max : x);
        return x;
    }
    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT T f_mod_python(const DEVICE& dev, T a, T b){
        return a - b * math::floor(dev, a / b);
    }

    template <typename DEVICE, typename T>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT T angle_normalize(const DEVICE& dev, T x){
        return f_mod_python(dev, (x + math::PI<T>), (2 * math::PI<T>)) - math::PI<T>;
    }
}
namespace backprop_tools{
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
        state.x = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
        state.y = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
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
        T cost = state.x * state.x + state.y * state.y;
        return -cost;
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
    // get_serialized_state is not generally required, it is just used in the WASM demonstration of the project page, where serialization is needed to go from the WASM runtime to the JavaScript UI
    template<typename DEVICE, typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T get_serialized_state(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State& state, typename DEVICE::index_t index){
        if(index == 0) {
            return state.theta;
        }
        else{
            return state.theta_dot;
        }
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Car<SPEC>& env, const typename rl::environments::Car<SPEC>::State state, RNG& rng){
        using T = typename SPEC::T;
        return false; //random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, (T)1, rng) > 0.9;
    }
}
#endif
