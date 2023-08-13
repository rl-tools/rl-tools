#ifndef BACKPROP_TOOLS_RL_ENVIRONMENTS_ACROBOT_OPERATIONS_GENERIC
#define BACKPROP_TOOLS_RL_ENVIRONMENTS_ACROBOT_OPERATIONS_GENERIC
#include "acrobot.h"
#include "../operations_generic.h"
// adapted from (and tested agains) https://github.com/Farama-Foundation/Gymnasium/blob/v0.28.1/gymnasium/envs/classic_control/acrobot.py
namespace backprop_tools::rl::environments::acrobot {
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
    template<typename DEVICE, typename SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT static void sample_initial_state(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, typename rl::environments::Acrobot<SPEC>::State& state, RNG& rng){
        state.theta_0     = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
        state.theta_1     = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
        state.theta_0_dot = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
        state.theta_1_dot = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), -0.1, 0.1, rng);
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, typename rl::environments::Acrobot<SPEC>::State& state){
        state.theta_0     = 0;
        state.theta_1     = 0;
        state.theta_0_dot = 0;
        state.theta_1_dot = 0;
    }
    namespace rl::environments::acrobot{
        template <typename T, typename PARAMS>
        void dsdt(T state[4], T action, T d_state[4], const PARAMS& params){

            T m1 = params.LINK_MASS_1;
            T m2 = params.LINK_MASS_2;
            T l1 = params.LINK_LENGTH_1;
            T lc1 = params.LINK_COM_POS_1;
            T lc2 = params.LINK_COM_POS_2;
            T I1 = params.LINK_MOI;
            T I2 = params.LINK_MOI;
            T g = 9.8;
            T theta1 = state[0];
            T theta2 = state[1];
            T dtheta1 = state[2];
            T dtheta2 = state[3];
            T d1 = (
                    m1 * lc1 * lc1
                    + m2 * (l1*l1 + lc2*lc2 + 2 * l1 * lc2 * cos(theta2))
                    + I1
                    + I2
            );
            T d2 = m2 * (lc2*lc2 + l1 * lc2 * cos(theta2)) + I2;
            T phi2 = m2 * lc2 * g * cos(theta1 + theta2 - math::PI<T> / 2.0);
            T phi1 = (
                    -m2 * l1 * lc2 * dtheta2 * dtheta2 * sin(theta2)
                    - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2)
                    + (m1 * lc1 + m2 * l1) * g * cos(theta1 - math::PI<T> / 2)
                    + phi2
            );

            T ddtheta2 = (
                               action + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1*dtheta1 * sin(theta2) - phi2
                       ) / (m2 * lc2*lc2 + I2 - d2*d2 / d1);
            T ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;
            d_state[0] = dtheta1;
            d_state[1] = dtheta2;
            d_state[2] = ddtheta1;
            d_state[3] = ddtheta2;
        }

        template <typename T, typename PARAMS>
        void rk4(T state[4], T action, T next_state[4], T dt, const PARAMS& params){

            T k1[4], k2[4], k3[4], k4[4], y1[4], y2[4], y3[4];

            T dt2 = dt / 2.0;

            rl::environments::acrobot::dsdt(state, action, k1, params);
            for (int i = 0; i < 4; ++i){
                y1[i] = state[i] + dt2 * k1[i];
            }
            rl::environments::acrobot::dsdt(y1, action, k2, params);
            for (int i = 0; i < 4; ++i){
                y2[i] = state[i] + dt2 * k2[i];
            }
            rl::environments::acrobot::dsdt(y2, action, k3, params);
            for (int i = 0; i < 4; ++i){
                y3[i] = state[i] + dt * k3[i];
            }
            rl::environments::acrobot::dsdt(y3, action, k4, params);
            for (int i = 0; i < 4; ++i){
                next_state[i] = state[i] + dt / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT typename SPEC::T step(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, typename rl::environments::Acrobot<SPEC>::State& next_state, RNG& rng) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 1);
        using namespace rl::environments::acrobot;
        using T = typename SPEC::T;
        using PARAMS = typename SPEC::PARAMETERS;

        T state_flat[4] = {state.theta_0, state.theta_1, state.theta_0_dot, state.theta_1_dot};
        T next_state_flat[4];
        rl::environments::acrobot::rk4(state_flat, get(action, 0, 0), next_state_flat, PARAMS::dt, PARAMS{});

        next_state_flat[0] = angle_normalize(typename DEVICE::SPEC::MATH(), next_state_flat[0]);
        next_state_flat[1] = angle_normalize(typename DEVICE::SPEC::MATH(), next_state_flat[1]);
        next_state_flat[2] = math::clamp(    typename DEVICE::SPEC::MATH(), next_state_flat[2], -PARAMS::MAX_VEL_1, PARAMS::MAX_VEL_1);
        next_state_flat[3] = math::clamp(    typename DEVICE::SPEC::MATH(), next_state_flat[3], -PARAMS::MAX_VEL_2, PARAMS::MAX_VEL_2);

        next_state.theta_0 = next_state_flat[0];
        next_state.theta_1 = next_state_flat[1];
        next_state.theta_0_dot = next_state_flat[2];
        next_state.theta_1_dot = next_state_flat[3];

        return SPEC::PARAMETERS::dt;
}
template<typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T reward(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State& state, const Matrix<ACTION_SPEC>& action, const typename rl::environments::Acrobot<SPEC>::State& next_state, RNG& rng){
    using namespace rl::environments::acrobot;
    typedef typename SPEC::T T;
    T angle_norm = angle_normalize(typename DEVICE::SPEC::MATH(), state.theta);
    T u_normalised = get(action, 0, 0);
    T u = SPEC::PARAMETERS::max_torque * u_normalised;
    T costs = angle_norm * angle_norm + 0.1 * state.theta_dot * state.theta_dot + 0.001 * (u * u);
    return -costs;
}

template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 3);
    typedef typename SPEC::T T;
    set(observation, 0, 0, math::cos(typename DEVICE::SPEC::MATH(), state.theta));
    set(observation, 0, 1, math::sin(typename DEVICE::SPEC::MATH(), state.theta));
    set(observation, 0, 2, state.theta_dot);
}
template<typename DEVICE, typename SPEC, typename OBS_SPEC, typename RNG>
BACKPROP_TOOLS_FUNCTION_PLACEMENT static void observe_privileged(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State& state, Matrix<OBS_SPEC>& observation, RNG& rng){
    static_assert(OBS_SPEC::ROWS == 1);
    static_assert(OBS_SPEC::COLS == 3);
    observe(device, env, state, observation, rng);
}
// get_serialized_state is not generally required, it is just used in the WASM demonstration of the project page, where serialization is needed to go from the WASM runtime to the JavaScript UI
template<typename DEVICE, typename SPEC>
BACKPROP_TOOLS_FUNCTION_PLACEMENT static typename SPEC::T get_serialized_state(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State& state, typename DEVICE::index_t index){
if(index == 0) {
return state.theta;
}
else{
return state.theta_dot;
}
}
template<typename DEVICE, typename SPEC, typename RNG>
BACKPROP_TOOLS_FUNCTION_PLACEMENT static bool terminated(DEVICE& device, const rl::environments::Acrobot<SPEC>& env, const typename rl::environments::Acrobot<SPEC>::State state, RNG& rng){
    using T = typename SPEC::T;
    return false; //random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), (T)0, (T)1, rng) > 0.9;
}
}
#endif