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
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 5000;
    static constexpr bool CLOSED_FORM = false;

    using STATE_BASE      = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_LAA       = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
    using STATE_IMU       = l2f::StateIMU<T, TI, STATE_LAA>;
    using STATE_MAHONY    = l2f::StateMahony<l2f::StateMahonySpecification<T, TI, STATE_IMU>>;
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
    static constexpr T STATE_LIMIT_POSITION_X = 100000;
    static constexpr T STATE_LIMIT_POSITION_Y = 100000;
    static constexpr T STATE_LIMIT_POSITION_Z = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
    static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
};

using ENV_SPEC = l2f::Specification<T, TI, STATIC_PARAMETERS>;
using ENV = rlt::rl::environments::Multirotor<ENV_SPEC>;

template <typename E>
static void const_action(rlt::Matrix<rlt::matrix::Specification<T, TI, 1, E::ACTION_DIM>>& a, T v){
    for (TI i = 0; i < E::ACTION_DIM; i++) rlt::set(a, 0, i, v);
}

static void set_q_from_world_z(ENV::State& state, T x, T y, T z){
    if (z < -0.999999){
        state.q_estimate[0] = 0;
        state.q_estimate[1] = 1;
        state.q_estimate[2] = 0;
        state.q_estimate[3] = 0;
        return;
    }
    T w = 1 + z;
    T qx = y;
    T qy = -x;
    T qz = 0;
    T qn = std::sqrt(w*w + qx*qx + qy*qy + qz*qz);
    state.q_estimate[0] = w / qn;
    state.q_estimate[1] = qx / qn;
    state.q_estimate[2] = qy / qn;
    state.q_estimate[3] = qz / qn;
}

static void world_z_from_q(const ENV::State& state, T world_z[3]){
    l2f::quaternion_to_world_z_body<DEVICE, T>(state.q_estimate, world_z);
}

TEST(L2F_MAHONY, STATE_DIM){
    using STATE = STATIC_PARAMETERS::STATE_MAHONY;
    static_assert(STATE::REQUIRES_INTEGRATION == false);
    static_assert(STATE::DIM == 7 + STATIC_PARAMETERS::STATE_IMU::DIM);
    static_assert(STATIC_PARAMETERS::STATE_IMU::DIM == 12 + STATIC_PARAMETERS::STATE_LAA::DIM); // accelerometer(3)+bias(3) + gyro(3)+bias(3)
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

TEST(L2F_MAHONY, GYRO_BIAS_SAMPLES_UNIFORM_AND_HOLDS_WITH_ZERO_SIGMA){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)7);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro.error.bias.init_max = 0.02;
    params.imu.gyro.error.bias.tau = 0; // tau <= 0: random walk, but sigma == 0 holds the turn-on bias
    params.imu.gyro.error.bias.sigma = 0;
    rlt::sample_initial_state(device, env, params, state, rng);

    T sampled_bias[3];
    for(TI i = 0; i < 3; i++){
        sampled_bias[i] = state.gyro_bias[i];
        EXPECT_LE(std::abs(sampled_bias[i]), params.imu.gyro.error.bias.init_max);
    }

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, params.dynamics.hovering_throttle_relative * 2 - 1);

    for(TI step_i = 0; step_i < 8; step_i++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }
    for(TI i = 0; i < 3; i++){
        EXPECT_DOUBLE_EQ(state.gyro_bias[i], sampled_bias[i]);
    }
    rlt::free(device, action);
}

TEST(L2F_MAHONY, ANGULAR_VELOCITY_OBSERVATION_USES_GYRO_BIAS){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)8);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.mdp.observation_noise.angular_velocity = 0;
    rlt::initial_state(device, env, params, state);
    state.angular_velocity[0] = 1.0;
    state.angular_velocity[1] = -2.0;
    state.angular_velocity[2] = 3.0;
    state.gyro_bias[0] = 0.1;
    state.gyro_bias[1] = -0.2;
    state.gyro_bias[2] = 0.3;

    using OBS = l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>;
    using OBS_PRIV = l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecificationPrivileged<T, TI>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS::DIM>> obs;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_PRIV::DIM>> obs_priv;
    rlt::malloc(device, obs);
    rlt::malloc(device, obs_priv);

    rlt::observe(device, env, params, state, OBS{}, obs, rng);
    rlt::observe(device, env, params, state, OBS_PRIV{}, obs_priv, rng);

    for(TI i = 0; i < 3; i++){
        EXPECT_DOUBLE_EQ(rlt::get(obs, 0, i), state.angular_velocity[i] + state.gyro_bias[i]);
        EXPECT_DOUBLE_EQ(rlt::get(obs_priv, 0, i), state.angular_velocity[i]);
    }

    rlt::free(device, obs);
    rlt::free(device, obs_priv);
}

TEST(L2F_MAHONY, OBSERVATION_USES_QUATERNION_ESTIMATE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)4);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);
    T expected[3] = {0.2, -0.3, std::sqrt(1 - 0.2*0.2 - 0.3*0.3)};
    set_q_from_world_z(state, expected[0], expected[1], expected[2]);

    using OBS = l2f::observation::OrientationMahonyWorldZ<l2f::observation::OrientationMahonyWorldZSpecification<T, TI>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS::DIM>> obs;
    rlt::malloc(device, obs);
    rlt::observe(device, env, params, state, OBS{}, obs, rng);

    EXPECT_NEAR(rlt::get(obs, 0, 0), expected[0], 1e-12);
    EXPECT_NEAR(rlt::get(obs, 0, 1), expected[1], 1e-12);
    EXPECT_NEAR(rlt::get(obs, 0, 2), expected[2], 1e-12);
    rlt::free(device, obs);
}

TEST(L2F_MAHONY, JSON_ROUND_TRIP_QUATERNION_STATE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)5);
    ENV env; ENV::Parameters params; ENV::State state, restored;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);
    state.q_estimate[0] = std::cos(0.25);
    state.q_estimate[1] = std::sin(0.25);
    state.q_estimate[2] = 0;
    state.q_estimate[3] = 0;
    state.bias_estimate[0] = 0.01;
    state.bias_estimate[1] = 0.02;
    state.bias_estimate[2] = 0.03;

    auto parsed = nlohmann::json::parse(rlt::json(device, env, params, state));
    rlt::from_json(device, env, params, parsed, restored);

    for(TI i = 0; i < 4; i++){
        EXPECT_NEAR(restored.q_estimate[i], state.q_estimate[i], 1e-6);
    }
    for(TI i = 0; i < 3; i++){
        EXPECT_NEAR(restored.bias_estimate[i], state.bias_estimate[i], 1e-6);
    }
}

TEST(L2F_MAHONY, FROM_JSON_ACCEPTS_LEGACY_REDUCED_STATE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)6);
    ENV env; ENV::Parameters params; ENV::State state, restored;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);

    auto parsed = nlohmann::json::parse(rlt::json(device, env, params, state));
    parsed.erase("q_estimate");
    parsed.erase("bias_estimate");
    parsed["world_z_body_estimate"] = {0, std::sin(0.5), std::cos(0.5)};
    parsed["gyro_bias_tangent"] = {0.01, 0.02, 0.03};
    T expected_bias[3] = {0.01, 0.02, 0.03};

    rlt::from_json(device, env, params, parsed, restored);

    T world_z[3];
    world_z_from_q(restored, world_z);
    EXPECT_NEAR(world_z[0], 0.0, 1e-12);
    EXPECT_NEAR(world_z[1], std::sin(0.5), 1e-12);
    EXPECT_NEAR(world_z[2], std::cos(0.5), 1e-12);
    for(TI i = 0; i < 3; i++){
        EXPECT_NEAR(restored.bias_estimate[i], expected_bias[i], 1e-12);
    }
}

// With zero gyro bias and no observation noise, the Mahony state should
// converge to the true world-Z-in-body direction in hover.
TEST(L2F_MAHONY, NO_BIAS_HOVER_CONVERGENCE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)1);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro.error.bias.init_max = 0;
    params.imu.gyro.error.bias.tau = 0;
    params.imu.gyro.error.bias.sigma = 0;
    rlt::initial_state(device, env, params, state);
    state.orientation[0] = 1; state.orientation[1] = 0; state.orientation[2] = 0; state.orientation[3] = 0;
    for (TI i = 0; i < 3; i++) { state.linear_velocity[i] = 0; state.angular_velocity[i] = 0; state.position[i] = 0; }

    set_q_from_world_z(state, 0, std::sin(0.5), std::cos(0.5));

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
    T world_z[3];
    world_z_from_q(state, world_z);
    EXPECT_NEAR(world_z[0], 0.0, 5e-2);
    EXPECT_NEAR(world_z[1], 0.0, 5e-2);
    EXPECT_NEAR(world_z[2], 1.0, 5e-2);
    rlt::free(device, action);
}

// With a constant gyro bias, only the gravity-tangent bias component is observable.
TEST(L2F_MAHONY, BIAS_REJECTION){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)2);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro.error.bias.init_max = 0;
    params.imu.gyro.error.bias.tau = 0;     // hold (constant bias)
    params.imu.gyro.error.bias.sigma = 0;
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
    TI n_steps = (TI)(600.0 / sim_dt);
    for (TI step = 0; step < n_steps; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }
    EXPECT_NEAR(state.bias_estimate[0], true_bias[0], 0.02)
        << "x: estimated " << state.bias_estimate[0] << " vs true " << true_bias[0];
    EXPECT_NEAR(state.bias_estimate[1], true_bias[1], 0.02)
        << "y: estimated " << state.bias_estimate[1] << " vs true " << true_bias[1];
    EXPECT_NEAR(state.bias_estimate[2], 0.0, 1e-6);
    T world_z[3];
    world_z_from_q(state, world_z);
    EXPECT_NEAR(world_z[0], 0.0, 5e-2);
    EXPECT_NEAR(world_z[1], 0.0, 5e-2);
    EXPECT_NEAR(world_z[2], 1.0, 5e-2);
    rlt::free(device, action);
}

TEST(L2F_MAHONY, PURE_YAW_BIAS_DOES_NOT_MOVE_WORLD_Z){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)3);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro.error.bias.init_max = 0;
    params.imu.gyro.error.bias.tau = 0;
    params.imu.gyro.error.bias.sigma = 0;
    params.mdp.observation_noise.angular_velocity = 0;
    params.mdp.observation_noise.imu_acceleration = 0;
    rlt::initial_state(device, env, params, state);
    state.gyro_bias[2] = 0.05;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    T hover_normalized = params.dynamics.hovering_throttle_relative * 2 - 1;
    const_action<ENV>(action, hover_normalized);

    TI n_steps = (TI)(60.0 / params.integration.dt);
    for (TI step = 0; step < n_steps; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }
    T world_z[3];
    world_z_from_q(state, world_z);
    EXPECT_NEAR(world_z[0], 0.0, 1e-6);
    EXPECT_NEAR(world_z[1], 0.0, 1e-6);
    EXPECT_NEAR(world_z[2], 1.0, 1e-6);
    EXPECT_NEAR(state.bias_estimate[0], 0.0, 1e-6);
    EXPECT_NEAR(state.bias_estimate[1], 0.0, 1e-6);
    EXPECT_NEAR(state.bias_estimate[2], 0.0, 1e-6);
    rlt::free(device, action);
}
