#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;

constexpr TI ACTION_HISTORY_LENGTH = 4;

struct STATIC_PARAMETERS{
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr bool CLOSED_FORM = false;

    using STATE_BASE_INNER = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_GYRO_BIAS  = l2f::StateGyroBias<l2f::StateGyroBiasSpecification<T, TI, STATE_BASE_INNER>>;
    using STATE_IMU        = l2f::StateIMU<l2f::StateIMUSpecification<T, TI, STATE_GYRO_BIAS>>;
    using STATE_TYPE       = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_IMU>>>>;

    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
        l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
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
using ACTION = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>>;

static void zero_noise(ENV::Parameters& params){
    params.mdp.observation_noise.imu_acceleration = 0;
    params.mdp.observation_noise.angular_velocity = 0;
    params.imu.gyro_bias.init_max = 0;
    params.imu.gyro_bias.tau = 0;
    params.imu.gyro_bias.sigma = 0;
    params.disturbances.random_force.mean = 0;
    params.disturbances.random_force.std = 0;
    params.disturbances.random_torque.mean = 0;
    params.disturbances.random_torque.std = 0;
}

static void hover_action(const ENV::Parameters& params, ACTION& action){
    for (TI i = 0; i < ENV::ACTION_DIM; i++){
        rlt::set(action, 0, i, params.dynamics.hovering_throttle_relative * 2 - 1);
    }
}

static void expected_measurement(const ENV::Parameters& params, const ENV::State& state, const ENV::State& next_state, T accelerometer[3], T gyroscope[3]){
    T conjugate_orientation[4] = {next_state.orientation[0], -next_state.orientation[1], -next_state.orientation[2], -next_state.orientation[3]};
    T acceleration_global[3];
    for (TI i = 0; i < 3; i++){
        acceleration_global[i] = (next_state.linear_velocity[i] - state.linear_velocity[i]) / params.integration.dt - params.dynamics.gravity[i];
    }
    DEVICE device;
    l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate_orientation, acceleration_global, accelerometer);
    for (TI i = 0; i < 3; i++){
        gyroscope[i] = next_state.angular_velocity[i] + next_state.gyro_bias[i];
    }
}

TEST(L2F_IMU_MEASUREMENT, STATE_DIM_AND_FLAGS){
    using STATE = STATIC_PARAMETERS::STATE_IMU;
    static_assert(STATE::REQUIRES_INTEGRATION == false);
    static_assert(STATE::DIM == 6 + STATIC_PARAMETERS::STATE_GYRO_BIAS::DIM);
    SUCCEED();
}

TEST(L2F_IMU_MEASUREMENT, INITIAL_STATE_ZEROES_MEASUREMENT){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    for (TI i = 0; i < 3; i++){
        state.imu_accelerometer[i] = 1234.5;
        state.imu_gyroscope[i] = 1234.5;
    }
    rlt::initial_state(device, env, params, state);
    for (TI i = 0; i < 3; i++){
        EXPECT_EQ(state.imu_accelerometer[i], 0.0);
        EXPECT_EQ(state.imu_gyroscope[i], 0.0);
    }
    rlt::sample_initial_state(device, env, params, state, rng);
    for (TI i = 0; i < 3; i++){
        EXPECT_EQ(state.imu_accelerometer[i], 0.0);
        EXPECT_EQ(state.imu_gyroscope[i], 0.0);
    }
}

TEST(L2F_IMU_MEASUREMENT, HOVER_SPECIFIC_FORCE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    zero_noise(params);
    rlt::initial_state(device, env, params, state);
    ACTION action;
    rlt::malloc(device, action);
    hover_action(params, action);
    rlt::step(device, env, params, state, action, next_state, rng);
    // sign/frame anchor: near hover the specific force is +1g on body z (up); the tolerance
    // absorbs the hovering_throttle_relative vs thrust-curve calibration mismatch of the model
    EXPECT_NEAR(next_state.imu_accelerometer[0], 0.0, 1e-1);
    EXPECT_NEAR(next_state.imu_accelerometer[1], 0.0, 1e-1);
    EXPECT_GT(next_state.imu_accelerometer[2], 7.0);
    EXPECT_LT(next_state.imu_accelerometer[2], 12.0);
    for (TI i = 0; i < 3; i++){
        EXPECT_EQ(next_state.imu_gyroscope[i], next_state.angular_velocity[i]);
    }
    rlt::free(device, action);
}

TEST(L2F_IMU_MEASUREMENT, FINITE_DIFFERENCE_CONSISTENCY){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)1);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    zero_noise(params);
    rlt::sample_initial_state(device, env, params, state, rng);
    ACTION action;
    rlt::malloc(device, action);
    for (TI step_i = 0; step_i < 100; step_i++){
        for (TI i = 0; i < ENV::ACTION_DIM; i++){
            rlt::set(action, 0, i, rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng));
        }
        rlt::step(device, env, params, state, action, next_state, rng);
        T accelerometer[3], gyroscope[3];
        expected_measurement(params, state, next_state, accelerometer, gyroscope);
        for (TI i = 0; i < 3; i++){
            EXPECT_NEAR(next_state.imu_accelerometer[i], accelerometer[i], 1e-12) << "step " << step_i << " axis " << i;
            EXPECT_NEAR(next_state.imu_gyroscope[i], gyroscope[i], 1e-12) << "step " << step_i << " axis " << i;
        }
        state = next_state;
    }
    rlt::free(device, action);
}

TEST(L2F_IMU_MEASUREMENT, GYRO_BIAS_PLUMBING){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)2);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    zero_noise(params);
    params.imu.gyro_bias.init_max = 0.05;
    params.imu.gyro_bias.tau = 0; // hold the turn-on bias
    rlt::sample_initial_state(device, env, params, state, rng);
    bool nonzero_bias = false;
    for (TI i = 0; i < 3; i++){
        nonzero_bias = nonzero_bias || state.gyro_bias[i] != 0;
        EXPECT_LE(std::abs(state.gyro_bias[i]), 0.05);
    }
    EXPECT_TRUE(nonzero_bias);
    ACTION action;
    rlt::malloc(device, action);
    hover_action(params, action);
    for (TI step_i = 0; step_i < 10; step_i++){
        rlt::step(device, env, params, state, action, next_state, rng);
        for (TI i = 0; i < 3; i++){
            EXPECT_EQ(next_state.gyro_bias[i], state.gyro_bias[i]);
            EXPECT_NEAR(next_state.imu_gyroscope[i] - next_state.angular_velocity[i], next_state.gyro_bias[i], 1e-15);
        }
        state = next_state;
    }
    rlt::free(device, action);
}

TEST(L2F_IMU_MEASUREMENT, NOISE_STD_SANITY){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)3);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    zero_noise(params);
    constexpr T ACCELEROMETER_NOISE_STD = 0.28;
    constexpr T GYRO_NOISE_STD = 0.07;
    params.mdp.observation_noise.imu_acceleration = ACCELEROMETER_NOISE_STD;
    params.mdp.observation_noise.angular_velocity = GYRO_NOISE_STD;
    rlt::initial_state(device, env, params, state);
    ACTION action;
    rlt::malloc(device, action);
    hover_action(params, action);
    constexpr TI N_SAMPLES = 2000;
    T accelerometer_residual_sq_sum = 0;
    T gyroscope_residual_sq_sum = 0;
    for (TI step_i = 0; step_i < N_SAMPLES; step_i++){
        rlt::step(device, env, params, state, action, next_state, rng);
        T accelerometer[3], gyroscope[3];
        expected_measurement(params, state, next_state, accelerometer, gyroscope);
        for (TI i = 0; i < 3; i++){
            T accelerometer_residual = next_state.imu_accelerometer[i] - accelerometer[i];
            T gyroscope_residual = next_state.imu_gyroscope[i] - gyroscope[i];
            accelerometer_residual_sq_sum += accelerometer_residual * accelerometer_residual;
            gyroscope_residual_sq_sum += gyroscope_residual * gyroscope_residual;
        }
        state = next_state;
    }
    T accelerometer_std = std::sqrt(accelerometer_residual_sq_sum / (N_SAMPLES * 3));
    T gyroscope_std = std::sqrt(gyroscope_residual_sq_sum / (N_SAMPLES * 3));
    EXPECT_NEAR(accelerometer_std, ACCELEROMETER_NOISE_STD, ACCELEROMETER_NOISE_STD * 0.3);
    EXPECT_NEAR(gyroscope_std, GYRO_NOISE_STD, GYRO_NOISE_STD * 0.3);
    rlt::free(device, action);
}

TEST(L2F_IMU_MEASUREMENT, DETERMINISM){
    ENV::State states[2];
    for (TI run_i = 0; run_i < 2; run_i++){
        DEVICE device; RNG rng;
        rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)4);
        ENV env; ENV::Parameters params; ENV::State state, next_state;
        rlt::malloc(device, env); rlt::init(device, env);
        rlt::sample_initial_parameters(device, env, params, rng);
        params.mdp.observation_noise.imu_acceleration = 0.28;
        params.mdp.observation_noise.angular_velocity = 0.07;
        params.imu.gyro_bias.init_max = 0.02;
        params.imu.gyro_bias.tau = 500;
        params.imu.gyro_bias.sigma = 0.01;
        rlt::sample_initial_state(device, env, params, state, rng);
        ACTION action;
        rlt::malloc(device, action);
        for (TI step_i = 0; step_i < 50; step_i++){
            for (TI i = 0; i < ENV::ACTION_DIM; i++){
                rlt::set(action, 0, i, rlt::random::uniform_real_distribution(device.random, (T)-1, (T)1, rng));
            }
            rlt::step(device, env, params, state, action, next_state, rng);
            state = next_state;
        }
        states[run_i] = state;
        rlt::free(device, action);
    }
    for (TI i = 0; i < 3; i++){
        EXPECT_EQ(states[0].imu_accelerometer[i], states[1].imu_accelerometer[i]);
        EXPECT_EQ(states[0].imu_gyroscope[i], states[1].imu_gyroscope[i]);
        EXPECT_EQ(states[0].gyro_bias[i], states[1].gyro_bias[i]);
    }
}

TEST(L2F_IMU_MEASUREMENT, JSON_ROUND_TRIP){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)5);
    ENV env; ENV::Parameters params; ENV::State state, next_state, restored;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    params.mdp.observation_noise.imu_acceleration = 0.28;
    params.mdp.observation_noise.angular_velocity = 0.07;
    rlt::sample_initial_state(device, env, params, state, rng);
    ACTION action;
    rlt::malloc(device, action);
    hover_action(params, action);
    rlt::step(device, env, params, state, action, next_state, rng);
    std::string json_string = rlt::json(device, env, params, next_state);
    rlt::initial_state(device, env, params, restored);
    rlt::from_json(device, env, params, nlohmann::json::parse(json_string), restored);
    for (TI i = 0; i < 3; i++){
        EXPECT_NEAR(restored.imu_accelerometer[i], next_state.imu_accelerometer[i], 1e-5);
        EXPECT_NEAR(restored.imu_gyroscope[i], next_state.imu_gyroscope[i], 1e-5);
    }
    rlt::free(device, action);
}
