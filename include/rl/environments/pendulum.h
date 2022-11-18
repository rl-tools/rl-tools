#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM

#include <stdbool.h>
#include "math.h"
#include <random>

template <typename T>
struct Pendulum {
    T g = 10;
    T max_speed = 8;
    T max_torque = 2;
    T dt = 0.05;
    T m = 1;
    T l = 1;
    T initial_state_min_angle = -M_PI;
    T initial_state_max_angle =  M_PI;
    T initial_state_min_speed = -1;
    T initial_state_max_speed =  1;
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
template <typename T, typename RNG>
T sample_initial_state(const Pendulum<T>& params, T state[2], RNG& rng){
    state[0] = std::uniform_real_distribution<T>(params.initial_state_min_angle, params.initial_state_max_angle)(rng);
    state[1] = std::uniform_real_distribution<T>(params.initial_state_min_speed, params.initial_state_max_speed)(rng);
}

template <typename T>
T step(const Pendulum<T>& params, const T state[2], const T action[1], T next_state[2]){
    T th = state[0];
    T thdot = state[1];
    T u_normalised = action[0];
    T u = params.max_torque * u_normalised;
    T g = params.g;
    T m = params.m;
    T l = params.l;
    T dt = params.dt;

    u = clip(u, -params.max_torque, params.max_torque);

    T newthdot = thdot + (3 * g / (2 * l) * std::sin(th) + 3.0 / (m * l * l) * u) * dt;
    newthdot = clip(newthdot, -params.max_speed, params.max_speed);
    T newth = th + newthdot * dt;

    T angle_norm = angle_normalize(th);
    T costs = angle_norm * angle_norm + 0.1 * thdot * thdot + 0.001 * (u * u);

    next_state[0] = newth;
    next_state[1] = newthdot;
    return -costs;
}

template <typename T>
void observe(const Pendulum<T>& pendulum, T state[2], T observation[3]){
    T th = state[0];
    T thdot = state[1];
    observation[0] = std::cos(th);
    observation[1] = std::cos(th);
    observation[2] = thdot;
}


#endif