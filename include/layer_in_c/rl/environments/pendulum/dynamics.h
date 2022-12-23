#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM
#include <random>
#include <layer_in_c/devices.h>
namespace layer_in_c::rl::environments::pendulum {
    template <typename T>
    inline T clip(T x, T min, T max){
        x = x < min ? min : (x > max ? max : x);
        return x;
    }
    template <typename T>
    T angle_normalize(T x){
        T temp = std::fmod(std::abs(x) + M_PI, 2*M_PI) - M_PI;
        temp = x > 0 ? temp : -temp;
        return temp;
    }
    template <typename T>
    struct DefaultParameters {
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
    template <typename T_T, typename T_PARAMETERS>
    struct Spec{
        typedef T_T T;
        typedef T_PARAMETERS PARAMETERS;
    };

    template <typename T>
    struct State{
        T theta;
        T theta_dot;
    };

}
namespace layer_in_c::rl::environments {
    template <typename DEVICE, typename SPEC>
    struct Pendulum: Environment{
        typedef pendulum::State<typename SPEC::T> State;
        typedef typename SPEC::PARAMETERS PARAMETERS;
        constexpr static uint32_t OBSERVATION_DIM = 3;
        constexpr static uint32_t ACTION_DIM = 1;
    };

}
namespace layer_in_c{
    template<typename SPEC, typename RNG>
    static void sample_initial_state(const rl::environments::Pendulum<devices::Generic, SPEC>& env, rl::environments::pendulum::State<typename SPEC::T>& state, RNG& rng){
        state.theta     = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_angle, SPEC::PARAMETERS::initial_state_max_angle)(rng);
        state.theta_dot = std::uniform_real_distribution<typename SPEC::T>(SPEC::PARAMETERS::initial_state_min_speed, SPEC::PARAMETERS::initial_state_max_speed)(rng);
    }
    template<typename SPEC>
    static typename SPEC::T step(const rl::environments::Pendulum<devices::Generic, SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, const typename SPEC::T action[1], rl::environments::pendulum::State<typename SPEC::T>& next_state) {
        using namespace rl::environments::pendulum;
        typedef typename SPEC::T T;
        typedef typename SPEC::PARAMETERS PARAMS;
        T u_normalised = action[0];
        T u = PARAMS::max_torque * u_normalised;
        T g = PARAMS::g;
        T m = PARAMS::m;
        T l = PARAMS::l;
        T dt = PARAMS::dt;

        u = clip(u, -PARAMS::max_torque, PARAMS::max_torque);

        T newthdot = state.theta_dot + (3 * g / (2 * l) * std::sin(state.theta) + 3.0 / (m * l * l) * u) * dt;
        newthdot = clip(newthdot, -PARAMS::max_speed, PARAMS::max_speed);
        T newth = state.theta + newthdot * dt;

        T angle_norm = angle_normalize(state.theta);
        T costs = angle_norm * angle_norm + 0.1 * state.theta_dot * state.theta_dot + 0.001 * (u * u);

        next_state.theta = newth;
        next_state.theta_dot = newthdot;
        return -costs;
    }
    template<typename SPEC>
    static void observe(const rl::environments::Pendulum<devices::Generic, SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, typename SPEC::T observation[3]){
        typedef typename SPEC::T T;
        observation[0] = std::cos(state.theta);
        observation[1] = std::sin(state.theta);
        observation[2] = state.theta_dot;
    }
    template<typename SPEC>
    static bool terminated(const rl::environments::Pendulum<devices::Generic, SPEC>& env, const typename rl::environments::pendulum::State<typename SPEC::T> state){
        return false;
    }
}








#endif