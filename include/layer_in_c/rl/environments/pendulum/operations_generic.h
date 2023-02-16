#ifndef LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_GENERIC
#define LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_OPERATIONS_GENERIC
#include "pendulum.h"
namespace layer_in_c::rl::environments::pendulum {
    template <typename T>
    __device__ __host__ T clip(T x, T min, T max){
        x = x < min ? min : (x > max ? max : x);
        return x;
    }
    template <typename DEVICE, typename T>
    __host__ __device__ T f_mod_python(const DEVICE& dev, T a, T b){
        printf("f_mod_python DEVICE: %d\n", DEVICE::DEVICE);
//        return a - b * math::floor(dev, a / b);
        return a - b * floorf(a / b);
    }

    template <typename DEVICE, typename T>
    __host__ __device__ T angle_normalize(const DEVICE& dev, T x){
        return f_mod_python(dev, (x + math::PI<T>), (2 * math::PI<T>)) - math::PI<T>;
    }
}
namespace layer_in_c{
    template<typename DEVICE, typename SPEC, typename RNG>
    __device__ static void sample_initial_state(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, typename rl::environments::pendulum::State<typename SPEC::T>& state, RNG& rng){
        state.theta     = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::initial_state_min_angle, SPEC::PARAMETERS::initial_state_max_angle, rng);
        state.theta_dot = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), SPEC::PARAMETERS::initial_state_min_speed, SPEC::PARAMETERS::initial_state_max_speed, rng);
    }
    template<typename DEVICE, typename SPEC>
    static void initial_state(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, typename rl::environments::pendulum::State<typename SPEC::T>& state){
        state.theta = -math::PI<typename SPEC::T>;
        state.theta_dot = 0;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    __device__ __host__ typename SPEC::T step(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, const Matrix<ACTION_SPEC>& action, rl::environments::pendulum::State<typename SPEC::T>& next_state) {
        static_assert(ACTION_SPEC::ROWS == 1);
        static_assert(ACTION_SPEC::COLS == 1);
        using namespace rl::environments::pendulum;
        typedef typename SPEC::T T;
        typedef typename SPEC::PARAMETERS PARAMS;
        T u_normalised = get(action, 0, 0);
        T u = PARAMS::max_torque * u_normalised;
        T g = PARAMS::g;
        T m = PARAMS::m;
        T l = PARAMS::l;
        T dt = PARAMS::dt;

        u = clip(u, -PARAMS::max_torque, PARAMS::max_torque);

        T sin_theta = sinf(state.theta); //math::sin(typename DEVICE::SPEC::MATH(), state.theta);
        T newthdot = state.theta_dot + (3 * g / (2 * l) * sin_theta + 3.0 / (m * l * l) * u) * dt;
        newthdot = clip(newthdot, -PARAMS::max_speed, PARAMS::max_speed);
        T newth = state.theta + newthdot * dt;

        next_state.theta = newth;
        next_state.theta_dot = newthdot;
        return SPEC::PARAMETERS::dt;
    }
    template<typename DEVICE, typename SPEC, typename ACTION_SPEC>
    __device__ __host__ static typename SPEC::T reward(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, const Matrix<ACTION_SPEC>& action, const rl::environments::pendulum::State<typename SPEC::T>& next_state){
        using namespace rl::environments::pendulum;
        typedef typename SPEC::T T;
        T angle_norm = angle_normalize(typename DEVICE::SPEC::MATH(), state.theta);
        T u_normalised = get(action, 0, 0);
        T u = SPEC::PARAMETERS::max_torque * u_normalised;
        T costs = angle_norm * angle_norm + 0.1 * state.theta_dot * state.theta_dot + 0.001 * (u * u);
        return -costs;
    }

    template<typename DEVICE, typename SPEC, typename OBS_SPEC>
    __host__ __device__ static void observe(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, const rl::environments::pendulum::State<typename SPEC::T>& state, Matrix<OBS_SPEC>& observation){
        static_assert(OBS_SPEC::ROWS == 1);
        static_assert(OBS_SPEC::COLS == 3);
        typedef typename SPEC::T T;
        set(observation, 0, 0, math::cos(typename DEVICE::SPEC::MATH(), state.theta));
        set(observation, 0, 1, math::sin(typename DEVICE::SPEC::MATH(), state.theta));
        set(observation, 0, 2, state.theta_dot);
    }
    template<typename DEVICE, typename SPEC>
    __host__ __device__ static bool terminated(DEVICE& device, const rl::environments::Pendulum<SPEC>& env, const typename rl::environments::pendulum::State<typename SPEC::T> state){
        return false;
    }
}
#endif
