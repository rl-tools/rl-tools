#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <gtest/gtest.h>

#include <cmath>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;

constexpr TI ACTION_HISTORY_LENGTH = 4;

struct STATIC_PARAMETERS {
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 5000;
    static constexpr bool CLOSED_FORM = false;

    using STATE_BASE      = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_LAA       = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
    using STATE_GYRO_BIAS = l2f::StateGyroBias<l2f::StateGyroBiasSpecification<T, TI, STATE_LAA>>;
    using STATE_MAHONY    = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_GYRO_BIAS>>;
    using STATE_TYPE      = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_MAHONY>>>>;

    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationWorldZ<l2f::observation::OrientationWorldZSpecification<T, TI,
        l2f::observation::OrientationMahonyWorldZ<l2f::observation::OrientationMahonyWorldZSpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
        l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
    using BASE = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;
    using PARAMETERS = typename BASE::PARAMETERS_TYPE;
    static constexpr auto PARAMETER_VALUES = BASE::nominal_parameters;
    static constexpr T STATE_LIMIT_POSITION = 100000;
    static constexpr T STATE_LIMIT_VELOCITY = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
};

using ENV_SPEC = l2f::Specification<T, TI, STATIC_PARAMETERS>;
using ENV = rlt::rl::environments::Multirotor<ENV_SPEC>;

template <typename E>
static void const_action(rlt::Matrix<rlt::matrix::Specification<T, TI, 1, E::ACTION_DIM>>& a, T v){
    for (TI i = 0; i < E::ACTION_DIM; i++) rlt::set(a, 0, i, v);
}

TEST(L2F_MAHONY, STATE_DIM){
    using STATE = STATIC_PARAMETERS::STATE_MAHONY;
    static_assert(STATE::REQUIRES_INTEGRATION == false);
    static_assert(STATE::DIM == 7 + STATIC_PARAMETERS::STATE_GYRO_BIAS::DIM);
    static_assert(STATIC_PARAMETERS::STATE_GYRO_BIAS::DIM == 3 + STATIC_PARAMETERS::STATE_LAA::DIM);
    SUCCEED();
}

TEST(L2F_MAHONY, INITIAL_STATE_IDENTITY){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);

    EXPECT_DOUBLE_EQ(state.q_estimate[0], 1.0);
    EXPECT_DOUBLE_EQ(state.q_estimate[1], 0.0);
    EXPECT_DOUBLE_EQ(state.q_estimate[2], 0.0);
    EXPECT_DOUBLE_EQ(state.q_estimate[3], 0.0);
    for (TI i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(state.bias_estimate[i], 0.0);
    for (TI i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(state.gyro_bias[i], 0.0);
}

// With zero gyro bias and no observation noise, the Mahony filter should
// converge to the true orientation when the drone is hovering at identity.
TEST(L2F_MAHONY, NO_BIAS_HOVER_CONVERGENCE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)1);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro_bias.init_max = 0;
    params.imu.gyro_bias.tau = 0;
    params.imu.gyro_bias.sigma = 0;
    rlt::initial_state(device, env, params, state);
    state.orientation[0] = 1; state.orientation[1] = 0; state.orientation[2] = 0; state.orientation[3] = 0;
    for (TI i = 0; i < 3; i++) { state.linear_velocity[i] = 0; state.angular_velocity[i] = 0; state.position[i] = 0; }

    // perturb the filter's initial estimate so we can watch convergence
    state.q_estimate[0] = std::cos(0.5 * 0.5);
    state.q_estimate[1] = std::sin(0.5 * 0.5);
    state.q_estimate[2] = 0;
    state.q_estimate[3] = 0;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    T hover_normalized = params.dynamics.hovering_throttle_relative * 2 - 1;
    const_action<ENV>(action, hover_normalized);

    T sim_dt = params.integration.dt;
    TI n_steps = (TI)(15.0 / sim_dt);
    for (TI step = 0; step < n_steps; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }
    EXPECT_NEAR(std::fabs(state.q_estimate[0]), 1.0, 5e-2);
    EXPECT_NEAR(std::fabs(state.q_estimate[1]), 0.0, 5e-2);
    rlt::free(device, action);
}

// With a constant gyro bias, the Mahony filter's bias_estimate should
// converge toward the true bias over time.
TEST(L2F_MAHONY, BIAS_REJECTION){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)2);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro_bias.init_max = 0;
    params.imu.gyro_bias.tau = 0;     // hold (constant bias)
    params.imu.gyro_bias.sigma = 0;
    params.mdp.observation_noise.angular_velocity = 0;
    params.mdp.observation_noise.imu_acceleration = 0;
    rlt::initial_state(device, env, params, state);
    state.orientation[0] = 1; state.orientation[1] = 0; state.orientation[2] = 0; state.orientation[3] = 0;
    for (TI i = 0; i < 3; i++) { state.linear_velocity[i] = 0; state.angular_velocity[i] = 0; state.position[i] = 0; }

    T true_bias[3] = {0.05, -0.03, 0.02};
    for (TI i = 0; i < 3; i++) state.gyro_bias[i] = true_bias[i];

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    T hover_normalized = params.dynamics.hovering_throttle_relative * 2 - 1;
    const_action<ENV>(action, hover_normalized);

    T sim_dt = params.integration.dt;
    TI n_steps = (TI)(60.0 / sim_dt);
    for (TI step = 0; step < n_steps; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }
    // Mahony observes only the gravity direction (2 dof), so yaw bias is unobservable.
    // Check x and y components of the bias only.
    EXPECT_NEAR(state.bias_estimate[0], true_bias[0], 0.02)
        << "x: estimated " << state.bias_estimate[0] << " vs true " << true_bias[0];
    EXPECT_NEAR(state.bias_estimate[1], true_bias[1], 0.02)
        << "y: estimated " << state.bias_estimate[1] << " vs true " << true_bias[1];
    rlt::free(device, action);
}
