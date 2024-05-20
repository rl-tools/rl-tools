#include <rl_tools/operations/cpu.h>
#include <gtest/gtest.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environment_wrappers/scale_observations/operations_generic.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
TEST(RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_SCALE_OBSERVATIONS, IDENTITY_SCALING){
    using SCALE_OBSERVATIONS_WRAPPER_SPEC = rlt::rl::environment_wrappers::scale_observations::Specification<T, TI>;
    using WRAPPED_ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_OBSERVATIONS_WRAPPER_SPEC, ENVIRONMENT>;
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::State state;
    WRAPPED_ENVIRONMENT wrapped_env;
    WRAPPED_ENVIRONMENT::State wrapped_state;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{});
    auto wrapped_rng = rng;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, WRAPPED_ENVIRONMENT::OBSERVATION_DIM>> wrapped_observation;
    rlt::initial_state(device, env, state);
    rlt::initial_state(device, wrapped_env, wrapped_state);
    rlt::observe(device, env, state, observation, rng);
    rlt::observe(device, wrapped_env, wrapped_state, wrapped_observation, wrapped_rng);
    T diff = rlt::abs_diff(device, observation, wrapped_observation);
    ASSERT_LT(diff, 1e-10);
}

struct SCALE_BY_10_OBSERVATIONS_WRAPPER_SPEC: rlt::rl::environment_wrappers::scale_observations::Specification<T, TI>{
    static constexpr T SCALE = 10;
};

TEST(RL_TOOLS_RL_ENVIRONMENT_WRAPPERS_SCALE_OBSERVATIONS, ACTUAL_SCALING){
    using WRAPPED_ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_BY_10_OBSERVATIONS_WRAPPER_SPEC, ENVIRONMENT>;
    DEVICE device;
    ENVIRONMENT env;
    ENVIRONMENT::State state;
    WRAPPED_ENVIRONMENT wrapped_env;
    WRAPPED_ENVIRONMENT::State wrapped_state;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{});
    auto wrapped_rng = rng;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observation;
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, WRAPPED_ENVIRONMENT::OBSERVATION_DIM>> wrapped_observation;
    rlt::initial_state(device, env, state);
    rlt::initial_state(device, wrapped_env, wrapped_state);
    rlt::observe(device, env, state, observation, rng);
    rlt::observe(device, wrapped_env, wrapped_state, wrapped_observation, wrapped_rng);
    rlt::multiply_all(device, wrapped_observation, 1/SCALE_BY_10_OBSERVATIONS_WRAPPER_SPEC::SCALE);
    T diff = rlt::abs_diff(device, observation, wrapped_observation);
    ASSERT_LT(diff, 1e-10);
}
