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
    static_assert(STATE::DIM == 6 + STATIC_PARAMETERS::STATE_GYRO_BIAS::DIM);
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

    EXPECT_DOUBLE_EQ(state.world_z_body_estimate[0], 0.0);
    EXPECT_DOUBLE_EQ(state.world_z_body_estimate[1], 0.0);
    EXPECT_DOUBLE_EQ(state.world_z_body_estimate[2], 1.0);
    for (TI i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(state.gyro_bias_tangent[i], 0.0);
    for (TI i = 0; i < 3; i++) EXPECT_DOUBLE_EQ(state.gyro_bias[i], 0.0);
}

TEST(L2F_MAHONY, OBSERVATION_USES_REDUCED_ATTITUDE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)4);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);
    state.world_z_body_estimate[0] = 0.2;
    state.world_z_body_estimate[1] = -0.3;
    state.world_z_body_estimate[2] = std::sqrt(1 - 0.2*0.2 - 0.3*0.3);

    using OBS = l2f::observation::OrientationMahonyWorldZ<l2f::observation::OrientationMahonyWorldZSpecification<T, TI>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS::DIM>> obs;
    rlt::malloc(device, obs);
    rlt::observe(device, env, params, state, OBS{}, obs, rng);

    EXPECT_NEAR(rlt::get(obs, 0, 0), state.world_z_body_estimate[0], 1e-12);
    EXPECT_NEAR(rlt::get(obs, 0, 1), state.world_z_body_estimate[1], 1e-12);
    EXPECT_NEAR(rlt::get(obs, 0, 2), state.world_z_body_estimate[2], 1e-12);
    rlt::free(device, obs);
}

TEST(L2F_MAHONY, JSON_ROUND_TRIP_REDUCED_STATE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)5);
    ENV env; ENV::Parameters params; ENV::State state, restored;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);
    state.world_z_body_estimate[0] = 0.2;
    state.world_z_body_estimate[1] = -0.3;
    state.world_z_body_estimate[2] = std::sqrt(1 - 0.2*0.2 - 0.3*0.3);
    state.gyro_bias_tangent[0] = 0.01;
    state.gyro_bias_tangent[1] = 0.02;
    state.gyro_bias_tangent[2] = -(state.gyro_bias_tangent[0]*state.world_z_body_estimate[0] + state.gyro_bias_tangent[1]*state.world_z_body_estimate[1])/state.world_z_body_estimate[2];

    auto parsed = nlohmann::json::parse(rlt::json(device, env, params, state));
    rlt::from_json(device, env, params, parsed, restored);

    for(TI i = 0; i < 3; i++){
        EXPECT_NEAR(restored.world_z_body_estimate[i], state.world_z_body_estimate[i], 1e-6);
        EXPECT_NEAR(restored.gyro_bias_tangent[i], state.gyro_bias_tangent[i], 1e-6);
    }
    T dot = 0;
    for(TI i = 0; i < 3; i++){
        dot += restored.gyro_bias_tangent[i] * restored.world_z_body_estimate[i];
    }
    EXPECT_NEAR(dot, 0.0, 1e-12);
}

TEST(L2F_MAHONY, FROM_JSON_ACCEPTS_LEGACY_QUATERNION_STATE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)6);
    ENV env; ENV::Parameters params; ENV::State state, restored;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);

    auto parsed = nlohmann::json::parse(rlt::json(device, env, params, state));
    parsed.erase("world_z_body_estimate");
    parsed.erase("gyro_bias_tangent");
    parsed["q_estimate"] = {std::cos(0.25), std::sin(0.25), 0, 0};
    parsed["bias_estimate"] = {0.01, 0.02, 0.03};

    rlt::from_json(device, env, params, parsed, restored);

    EXPECT_NEAR(restored.world_z_body_estimate[0], 0.0, 1e-12);
    EXPECT_NEAR(restored.world_z_body_estimate[1], std::sin(0.5), 1e-12);
    EXPECT_NEAR(restored.world_z_body_estimate[2], std::cos(0.5), 1e-12);
    T dot = 0;
    for(TI i = 0; i < 3; i++){
        dot += restored.gyro_bias_tangent[i] * restored.world_z_body_estimate[i];
    }
    EXPECT_NEAR(dot, 0.0, 1e-12);
}

// With zero gyro bias and no observation noise, the reduced Mahony state should
// converge to the true world-Z-in-body direction in hover.
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

    state.world_z_body_estimate[0] = 0;
    state.world_z_body_estimate[1] = std::sin(0.5);
    state.world_z_body_estimate[2] = std::cos(0.5);

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
    EXPECT_NEAR(state.world_z_body_estimate[0], 0.0, 5e-2);
    EXPECT_NEAR(state.world_z_body_estimate[1], 0.0, 5e-2);
    EXPECT_NEAR(state.world_z_body_estimate[2], 1.0, 5e-2);
    rlt::free(device, action);
}

// With a constant gyro bias, only the gravity-tangent bias component is observable.
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
    EXPECT_NEAR(state.gyro_bias_tangent[0], true_bias[0], 0.02)
        << "x: estimated " << state.gyro_bias_tangent[0] << " vs true " << true_bias[0];
    EXPECT_NEAR(state.gyro_bias_tangent[1], true_bias[1], 0.02)
        << "y: estimated " << state.gyro_bias_tangent[1] << " vs true " << true_bias[1];
    EXPECT_NEAR(state.gyro_bias_tangent[2], 0.0, 1e-6);
    EXPECT_NEAR(state.world_z_body_estimate[0], 0.0, 5e-2);
    EXPECT_NEAR(state.world_z_body_estimate[1], 0.0, 5e-2);
    EXPECT_NEAR(state.world_z_body_estimate[2], 1.0, 5e-2);
    rlt::free(device, action);
}

TEST(L2F_MAHONY, PURE_YAW_BIAS_DOES_NOT_MOVE_REDUCED_ATTITUDE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)3);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.imu.gyro_bias.init_max = 0;
    params.imu.gyro_bias.tau = 0;
    params.imu.gyro_bias.sigma = 0;
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
    EXPECT_NEAR(state.world_z_body_estimate[0], 0.0, 1e-6);
    EXPECT_NEAR(state.world_z_body_estimate[1], 0.0, 1e-6);
    EXPECT_NEAR(state.world_z_body_estimate[2], 1.0, 1e-6);
    EXPECT_NEAR(state.gyro_bias_tangent[0], 0.0, 1e-6);
    EXPECT_NEAR(state.gyro_bias_tangent[1], 0.0, 1e-6);
    EXPECT_NEAR(state.gyro_bias_tangent[2], 0.0, 1e-6);
    rlt::free(device, action);
}
