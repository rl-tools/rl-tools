#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM

#include <stdbool.h>
#include "math.h"
#include <random>
template <typename T>
struct DefaultPendulumParams {
    constexpr static T g = 10;
    constexpr static T max_speed = 8;
    constexpr static T max_torque = 2;
    constexpr static T dt = 0.05;
    constexpr static T m = 1;
    constexpr static T l = 1;
    constexpr static T initial_state_min_angle = -M_PI;
    constexpr static T initial_state_max_angle = M_PI;
    constexpr static T initial_state_min_speed = -1;
    constexpr static T initial_state_max_speed = 1;
};

template <typename T>
inline T clip(T x, T min, T max){
    x = x < min ? min : (x > max ? max : x);
    return x;
}

template <typename T>
inline T angle_normalize(T x){
    return std::fmod((x + M_PI), (2 * M_PI)) - M_PI;
}

template <typename T, typename PARAMS>
struct Pendulum {
    constexpr static uint32_t STATE_DIM = 2;
    constexpr static uint32_t OBSERVATION_DIM = 3;
    constexpr static uint32_t ACTION_DIM = 1;
    template<typename RNG>
    static T sample_initial_state(T state[2], RNG& rng){
        state[0] = std::uniform_real_distribution<T>(PARAMS::initial_state_min_angle, PARAMS::initial_state_max_angle)(rng);
        state[1] = std::uniform_real_distribution<T>(PARAMS::initial_state_min_speed, PARAMS::initial_state_max_speed)(rng);
    }
    static T step(const T state[2], const T action[1], T next_state[2]) {
        T th = state[0];
        T thdot = state[1];
        T u_normalised = action[0];
        T u = PARAMS::max_torque * u_normalised;
        T g = PARAMS::g;
        T m = PARAMS::m;
        T l = PARAMS::l;
        T dt = PARAMS::dt;

        u = clip(u, -PARAMS::max_torque, PARAMS::max_torque);

        T newthdot = thdot + (3 * g / (2 * l) * std::sin(th) + 3.0 / (m * l * l) * u) * dt;
        newthdot = clip(newthdot, -PARAMS::max_speed, PARAMS::max_speed);
        T newth = th + newthdot * dt;

        T angle_norm = angle_normalize(th);
        T costs = angle_norm * angle_norm + 0.1 * thdot * thdot + 0.001 * (u * u);

        next_state[0] = newth;
        next_state[1] = newthdot;
        return -costs;
    }
    static void observe(T state[2], T observation[3]){
        T th = state[0];
        T thdot = state[1];
        observation[0] = std::cos(th);
        observation[1] = std::cos(th);
        observation[2] = thdot;
    }
};







#endif