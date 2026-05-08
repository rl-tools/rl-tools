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
constexpr TI ACCEL_HISTORY_LENGTH = 5;

template <bool PRIVILEGED_NOISE = false>
struct STATIC_PARAMETERS_TEMPLATE{
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr bool CLOSED_FORM = false;

    using STATE_BASE_INNER = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_BASE_LAA   = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE_INNER>>;
    using STATE_BASE_HIST  = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACCEL_HISTORY_LENGTH, STATE_BASE_LAA>>;
    using STATE_TYPE       = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_BASE_HIST>>>>;

    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
        l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
        l2f::observation::LinearAccelerationBodyFrame<l2f::observation::LinearAccelerationBodyFrameSpecification<T, TI,
        l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecification<T, TI, ACCEL_HISTORY_LENGTH,
        l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = PRIVILEGED_NOISE;
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

using STATIC_PARAMETERS = STATIC_PARAMETERS_TEMPLATE<false>;
using ENV_SPEC = l2f::Specification<T, TI, STATIC_PARAMETERS>;
using ENV = rlt::rl::environments::Multirotor<ENV_SPEC>;

template <typename E>
static void zero_action(rlt::Matrix<rlt::matrix::Specification<T, TI, 1, E::ACTION_DIM>>& a){
    for (TI i = 0; i < E::ACTION_DIM; i++) rlt::set(a, 0, i, (T)0);
}
template <typename E>
static void const_action(rlt::Matrix<rlt::matrix::Specification<T, TI, 1, E::ACTION_DIM>>& a, T v){
    for (TI i = 0; i < E::ACTION_DIM; i++) rlt::set(a, 0, i, v);
}

template <typename E>
static auto& accel_layer(typename E::State& s){
    using LAYER = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, ACCEL_HISTORY_LENGTH, typename STATIC_PARAMETERS::STATE_BASE_LAA>>;
    return static_cast<LAYER&>(s);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, STATE_DIM_AND_FLAGS){
    using STATE = STATIC_PARAMETERS::STATE_BASE_HIST;
    static_assert(STATE::REQUIRES_INTEGRATION == false);
    static_assert(STATE::HISTORY_LENGTH == ACCEL_HISTORY_LENGTH);
    static_assert(STATE::ACCELERATION_DIM == 3);
    static_assert(STATE::DIM == ACCEL_HISTORY_LENGTH * 3 + STATIC_PARAMETERS::STATE_BASE_LAA::DIM);
    SUCCEED();
}

TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVATION_DIM){
    using OBS = STATIC_PARAMETERS::OBSERVATION_TYPE;
    using HIST_OBS = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecification<T, TI, ACCEL_HISTORY_LENGTH>>;
    static_assert(HIST_OBS::CURRENT_DIM == ACCEL_HISTORY_LENGTH * 3);
    static_assert(HIST_OBS::ACCELERATION_DIM == 3);
    static_assert(HIST_OBS::HISTORY_LENGTH == ACCEL_HISTORY_LENGTH);
    static_assert(OBS::DIM == 3 + 9 + 3 + 3 + 3 + ACCEL_HISTORY_LENGTH * 3 + ACTION_HISTORY_LENGTH * 4);
    SUCCEED();
}

TEST(L2F_LINEAR_ACCEL_HISTORY, INITIAL_STATE_ZEROES_BUFFER){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);

    auto& a_state = accel_layer<ENV>(state);
    a_state.acceleration_history_step = 99;
    for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
        for (TI d = 0; d < 3; d++)
            a_state.linear_acceleration_body_history[i][d] = 1234.5;

    rlt::initial_state(device, env, params, state);

    EXPECT_EQ(a_state.acceleration_history_step, 0u);
    for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
        for (TI d = 0; d < 3; d++)
            EXPECT_EQ(a_state.linear_acceleration_body_history[i][d], 0.0)
                << "step " << i << " dim " << d;
}

TEST(L2F_LINEAR_ACCEL_HISTORY, SAMPLE_INITIAL_STATE_ZEROES_BUFFER){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);

    for (TI trial = 0; trial < 5; trial++){
        auto& a_state = accel_layer<ENV>(state);
        a_state.acceleration_history_step = 99;
        for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
            for (TI d = 0; d < 3; d++)
                a_state.linear_acceleration_body_history[i][d] = 999.0;

        rlt::sample_initial_state(device, env, params, state, rng);

        EXPECT_EQ(a_state.acceleration_history_step, 0u);
        for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
            for (TI d = 0; d < 3; d++)
                EXPECT_EQ(a_state.linear_acceleration_body_history[i][d], 0.0);
    }
}

TEST(L2F_LINEAR_ACCEL_HISTORY, RING_BUFFER_STEP_ADVANCES){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.5);

    auto& a_state = accel_layer<ENV>(state);
    EXPECT_EQ(a_state.acceleration_history_step, 0u);

    for (TI step = 1; step <= 3 * ACCEL_HISTORY_LENGTH; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
        TI expected = step % ACCEL_HISTORY_LENGTH;
        EXPECT_EQ(a_state.acceleration_history_step, expected) << "step " << step;
    }
    rlt::free(device, action);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, POST_INTEGRATION_PUSHES_LATEST_AT_STEP_INDEX){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.5);

    auto& a_state = accel_layer<ENV>(state);
    EXPECT_EQ(a_state.acceleration_history_step, 0u);

    for (TI step = 0; step < ACCEL_HISTORY_LENGTH; step++){
        T v_before[3] = {state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2]};
        rlt::step(device, env, params, state, action, next_state, rng);

        T q_conj[4] = {next_state.orientation[0], -next_state.orientation[1], -next_state.orientation[2], -next_state.orientation[3]};
        T a_world[3];
        for (TI d = 0; d < 3; d++)
            a_world[d] = (next_state.linear_velocity[d] - v_before[d])/params.integration.dt - params.dynamics.gravity[d];
        T a_body_expected[3];
        l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conj, a_world, a_body_expected);

        TI write_idx = step;
        for (TI d = 0; d < 3; d++){
            EXPECT_NEAR(accel_layer<ENV>(next_state).linear_acceleration_body_history[write_idx][d], a_body_expected[d], 1e-9)
                << "step " << step << " dim " << d;
        }
        state = next_state;
    }
    rlt::free(device, action);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, RING_BUFFER_PRESERVES_OLD_ENTRIES_AFTER_WRAP){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.4);

    T snapshots[3 * ACCEL_HISTORY_LENGTH][3];
    for (TI step = 0; step < 3 * ACCEL_HISTORY_LENGTH; step++){
        T v_before[3] = {state.linear_velocity[0], state.linear_velocity[1], state.linear_velocity[2]};
        rlt::step(device, env, params, state, action, next_state, rng);

        T q_conj[4] = {next_state.orientation[0], -next_state.orientation[1], -next_state.orientation[2], -next_state.orientation[3]};
        T a_world[3];
        for (TI d = 0; d < 3; d++)
            a_world[d] = (next_state.linear_velocity[d] - v_before[d])/params.integration.dt - params.dynamics.gravity[d];
        l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conj, a_world, snapshots[step]);
        state = next_state;
    }

    auto& a_state = accel_layer<ENV>(state);
    TI total = 3 * ACCEL_HISTORY_LENGTH;
    for (TI age = 1; age <= ACCEL_HISTORY_LENGTH; age++){
        TI buf_index = (a_state.acceleration_history_step + ACCEL_HISTORY_LENGTH - age) % ACCEL_HISTORY_LENGTH;
        TI step_index = total - age;
        for (TI d = 0; d < 3; d++){
            EXPECT_NEAR(a_state.linear_acceleration_body_history[buf_index][d], snapshots[step_index][d], 1e-9)
                << "age " << age << " dim " << d;
        }
    }
    rlt::free(device, action);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVE_RETURNS_MOST_RECENT_FIRST){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.55);

    for (TI step = 0; step < 3 * ACCEL_HISTORY_LENGTH; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }

    using HIST_OBS = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecificationPrivileged<T, TI, ACCEL_HISTORY_LENGTH>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, HIST_OBS::CURRENT_DIM>> obs;
    rlt::malloc(device, obs);
    rlt::observe(device, env, params, state, HIST_OBS{}, obs, rng);

    auto& a_state = accel_layer<ENV>(state);
    TI cursor = a_state.acceleration_history_step == 0 ? ACCEL_HISTORY_LENGTH - 1 : a_state.acceleration_history_step - 1;
    for (TI step_i = 0; step_i < ACCEL_HISTORY_LENGTH; step_i++){
        for (TI d = 0; d < 3; d++){
            EXPECT_NEAR(rlt::get(obs, 0, step_i * 3 + d), a_state.linear_acceleration_body_history[cursor][d], 1e-12)
                << "step_i " << step_i << " dim " << d;
        }
        cursor = cursor == 0 ? ACCEL_HISTORY_LENGTH - 1 : cursor - 1;
    }
    rlt::free(device, action); rlt::free(device, obs);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVE_SUB_WINDOW_TAKES_MOST_RECENT_K){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.45);

    for (TI step = 0; step < 2 * ACCEL_HISTORY_LENGTH; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }

    constexpr TI SUB = 3;
    using SUB_OBS = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecificationPrivileged<T, TI, SUB>>;
    using FULL_OBS = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecificationPrivileged<T, TI, ACCEL_HISTORY_LENGTH>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, SUB_OBS::CURRENT_DIM>> obs_sub;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, FULL_OBS::CURRENT_DIM>> obs_full;
    rlt::malloc(device, obs_sub); rlt::malloc(device, obs_full);
    rlt::observe(device, env, params, state, SUB_OBS{}, obs_sub, rng);
    rlt::observe(device, env, params, state, FULL_OBS{}, obs_full, rng);

    for (TI i = 0; i < SUB * 3; i++){
        EXPECT_NEAR(rlt::get(obs_sub, 0, i), rlt::get(obs_full, 0, i), 1e-12) << "i " << i;
    }
    rlt::free(device, action); rlt::free(device, obs_sub); rlt::free(device, obs_full);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVE_NOISE_REPRODUCIBLE){
    DEVICE device;
    RNG rng_a, rng_b, rng_c, rng_setup;
    rlt::init(device);
    rlt::malloc(device, rng_a); rlt::malloc(device, rng_b); rlt::malloc(device, rng_c); rlt::malloc(device, rng_setup);
    rlt::init(device, rng_a, (TI)123); rlt::init(device, rng_b, (TI)123); rlt::init(device, rng_c, (TI)123);
    rlt::init(device, rng_setup, (TI)7);

    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng_setup);
    params.mdp.observation_noise.imu_acceleration = 0.5;
    rlt::sample_initial_state(device, env, params, state, rng_setup);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.5);
    for (TI step = 0; step < 6; step++){
        rlt::step(device, env, params, state, action, next_state, rng_setup);
        state = next_state;
    }

    using NOISY = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecification<T, TI, ACCEL_HISTORY_LENGTH>>;
    using PRIV  = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecificationPrivileged<T, TI, ACCEL_HISTORY_LENGTH>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, NOISY::CURRENT_DIM>> obs_a, obs_b, obs_priv;
    rlt::malloc(device, obs_a); rlt::malloc(device, obs_b); rlt::malloc(device, obs_priv);
    rlt::observe(device, env, params, state, NOISY{}, obs_a, rng_a);
    rlt::observe(device, env, params, state, NOISY{}, obs_b, rng_b);
    rlt::observe(device, env, params, state, PRIV{},  obs_priv, rng_c);

    bool any_diff_to_priv = false;
    for (TI i = 0; i < NOISY::CURRENT_DIM; i++){
        EXPECT_DOUBLE_EQ(rlt::get(obs_a, 0, i), rlt::get(obs_b, 0, i)) << "noise differs across identical RNGs at i=" << i;
        if (rlt::get(obs_a, 0, i) != rlt::get(obs_priv, 0, i)) any_diff_to_priv = true;
    }
    EXPECT_TRUE(any_diff_to_priv) << "non-privileged observation should have measurable noise vs. privileged";
    rlt::free(device, action); rlt::free(device, obs_a); rlt::free(device, obs_b); rlt::free(device, obs_priv);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVE_PRIVILEGED_MATCHES_STORED){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.5);
    for (TI step = 0; step < 4; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }

    using PRIV = l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecificationPrivileged<T, TI, ACCEL_HISTORY_LENGTH>>;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, PRIV::CURRENT_DIM>> obs;
    rlt::malloc(device, obs);
    rlt::observe(device, env, params, state, PRIV{}, obs, rng);

    auto& a_state = accel_layer<ENV>(state);
    TI cursor = a_state.acceleration_history_step == 0 ? ACCEL_HISTORY_LENGTH - 1 : a_state.acceleration_history_step - 1;
    for (TI step_i = 0; step_i < ACCEL_HISTORY_LENGTH; step_i++){
        for (TI d = 0; d < 3; d++){
            EXPECT_DOUBLE_EQ(rlt::get(obs, 0, step_i * 3 + d), a_state.linear_acceleration_body_history[cursor][d]);
        }
        cursor = cursor == 0 ? ACCEL_HISTORY_LENGTH - 1 : cursor - 1;
    }
    rlt::free(device, action); rlt::free(device, obs);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, JSON_ROUND_TRIP){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, state_recon, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.5);
    for (TI step = 0; step < 8; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }

    auto json_str = rlt::json(device, env, params, state);
    nlohmann::json parsed = nlohmann::json::parse(json_str);
    rlt::from_json(device, env, params, parsed, state_recon);
    auto json_str_recon = rlt::json(device, env, params, state_recon);
    EXPECT_EQ(json_str, json_str_recon);

    auto& a_orig  = accel_layer<ENV>(state);
    auto& a_recon = accel_layer<ENV>(state_recon);
    EXPECT_EQ(a_orig.acceleration_history_step, a_recon.acceleration_history_step);
    // lossless float serialization: must round-trip with bitwise equality
    for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
        for (TI d = 0; d < 3; d++)
            EXPECT_DOUBLE_EQ(a_orig.linear_acceleration_body_history[i][d], a_recon.linear_acceleration_body_history[i][d])
                << "i=" << i << " d=" << d;
    rlt::free(device, action);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, FROM_JSON_REJECTS_OOB_STEP){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, state_recon;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);
    auto json_str = rlt::json(device, env, params, state);
    nlohmann::json parsed = nlohmann::json::parse(json_str);
    parsed["acceleration_history_step"] = ACCEL_HISTORY_LENGTH;  // out-of-range
    EXPECT_THROW(rlt::from_json(device, env, params, parsed, state_recon), std::runtime_error);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, FROM_JSON_REJECTS_WRONG_BUFFER_SHAPE){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, state_recon;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);
    auto json_str = rlt::json(device, env, params, state);
    nlohmann::json parsed_short = nlohmann::json::parse(json_str);
    parsed_short["linear_acceleration_body_history"].erase(parsed_short["linear_acceleration_body_history"].size() - 1);
    EXPECT_THROW(rlt::from_json(device, env, params, parsed_short, state_recon), std::runtime_error);

    nlohmann::json parsed_narrow = nlohmann::json::parse(json_str);
    parsed_narrow["linear_acceleration_body_history"][0].erase(parsed_narrow["linear_acceleration_body_history"][0].size() - 1);
    EXPECT_THROW(rlt::from_json(device, env, params, parsed_narrow, state_recon), std::runtime_error);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, JSON_FLOAT_VALUES_ROUND_TRIP_BIT_EXACT){
    // Pin the lossless contract: arbitrary doubles must survive json+from_json
    // unchanged. Uses constructed-by-hand JSON to avoid any reliance on the
    // initial-state distribution.
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, state_recon;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    auto& a_state = accel_layer<ENV>(state);
    const T values[ACCEL_HISTORY_LENGTH * 3] = {
        1.0/3.0,
        std::nextafter(1.0, 2.0),
        -std::nextafter(0.0, 1.0),
        1.7976931348623157e+200,
        -1.234567890123456789e-7,
        1e-300,
        std::numeric_limits<T>::epsilon(),
        3.14159265358979323846,
        std::nextafter(2.5, 3.0),
        -2.71828182845904523536,
        std::numeric_limits<T>::lowest() / 2,
        std::numeric_limits<T>::max() / 2,
        0.1, 0.2, 0.30000000000000004,
    };
    for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++)
        for (TI d = 0; d < 3; d++)
            a_state.linear_acceleration_body_history[i][d] = values[i*3 + d];
    a_state.acceleration_history_step = ACCEL_HISTORY_LENGTH - 1;

    auto json_str = rlt::json(device, env, params, state);
    nlohmann::json parsed = nlohmann::json::parse(json_str);
    rlt::from_json(device, env, params, parsed, state_recon);

    auto& a_recon = accel_layer<ENV>(state_recon);
    EXPECT_EQ(a_recon.acceleration_history_step, ACCEL_HISTORY_LENGTH - 1);
    for (TI i = 0; i < ACCEL_HISTORY_LENGTH; i++){
        for (TI d = 0; d < 3; d++){
            T orig = a_state.linear_acceleration_body_history[i][d];
            T recv = a_recon.linear_acceleration_body_history[i][d];
            EXPECT_EQ(std::memcmp(&orig, &recv, sizeof(T)), 0)
                << "bit-exact round-trip failed at i=" << i << " d=" << d
                << " orig=" << orig << " recv=" << recv;
        }
    }
}

// Skipping a runtime _is_nan test: (a) the pre-existing _is_nan overloads in
// operations_generic/05_state_is_nan.h call an undefined is_nan for recursion
// (no top-level is_nan dispatch exists for state types), and (b) the project
// is compiled with -ffast-math which neutralises std::isnan. The buffer
// iteration shape is verified statically below.
TEST(L2F_LINEAR_ACCEL_HISTORY, BUFFER_LAYOUT_AND_FIELD_TYPES){
    using STATE = STATIC_PARAMETERS::STATE_BASE_HIST;
    STATE s{};
    static_assert(STATE::HISTORY_LENGTH == ACCEL_HISTORY_LENGTH);
    static_assert(STATE::HISTORY_MEM_LENGTH == ACCEL_HISTORY_LENGTH);
    static_assert(sizeof(s.linear_acceleration_body_history) == sizeof(T) * ACCEL_HISTORY_LENGTH * 3);
    static_assert(std::is_same_v<decltype(s.linear_acceleration_body_history[0][0]), T&>);
    static_assert(std::is_same_v<decltype(s.acceleration_history_step), TI>);
    SUCCEED();
}

TEST(L2F_LINEAR_ACCEL_HISTORY, ZERO_LENGTH_STATE_HAS_DUMMY_STORAGE){
    using INNER = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using ZERO  = l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, 0, INNER>>;
    static_assert(ZERO::HISTORY_LENGTH == 0);
    static_assert(ZERO::HISTORY_MEM_LENGTH == 1);  // dummy slot to avoid zero-length array
    static_assert(ZERO::DIM == INNER::DIM);  // contributes nothing to observation/state DIM
    ZERO z{};
    static_assert(sizeof(z.linear_acceleration_body_history) == sizeof(T) * 1 * 3);
    SUCCEED();
}

template <bool PRIVILEGED_NOISE>
struct STATIC_PARAMETERS_ZERO_TEMPLATE{
    static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
    static constexpr TI N_SUBSTEPS = 1;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    static constexpr bool CLOSED_FORM = false;
    using STATE_BASE_INNER = l2f::StateBase<l2f::StateSpecification<T, TI>>;
    using STATE_BASE_LAA   = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE_INNER>>;
    using STATE_TYPE       = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, 1, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI,
                              l2f::StateLinearAccelerationHistory<l2f::StateLinearAccelerationHistorySpecification<T, TI, 0, STATE_BASE_LAA>>>>>>;
    using OBSERVATION_TYPE =
        l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
        l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
        l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
        l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI,
        l2f::observation::LinearAccelerationBodyFrameHistory<l2f::observation::LinearAccelerationBodyFrameHistorySpecification<T, TI, 0,
        l2f::observation::ActionHistory<l2f::observation::ActionHistorySpecification<T, TI, 1>>>>>>>>>>>>;
    using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = PRIVILEGED_NOISE;
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
using ENV_ZERO = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, STATIC_PARAMETERS_ZERO_TEMPLATE<false>>>;

TEST(L2F_LINEAR_ACCEL_HISTORY, ZERO_LENGTH_STEPS_AND_OBSERVES_WITHOUT_UB){
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV_ZERO env; ENV_ZERO::Parameters params; ENV_ZERO::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::sample_initial_state(device, env, params, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV_ZERO::ACTION_DIM>> action;
    rlt::malloc(device, action);
    for (TI i = 0; i < ENV_ZERO::ACTION_DIM; i++) rlt::set(action, 0, i, (T)0.5);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV_ZERO::Observation::DIM>> obs;
    rlt::malloc(device, obs);
    for (TI step = 0; step < 4; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        rlt::observe(device, env, params, next_state, ENV_ZERO::Observation{}, obs, rng);
        state = next_state;
    }
    SUCCEED();  // pure compile + run smoke test for zero-length safety
}

TEST(L2F_LINEAR_ACCEL_HISTORY, HOVER_AT_IDENTITY_GIVES_GRAVITY_MAGNITUDE_ALONG_BODY_Z){
    // Sanity: at hover with identity orientation, IMU specific force should be
    // dominantly along body +z with magnitude ~|g|. We allow generous slack
    // because the multirotor at action = hovering_throttle isn't a perfectly
    // stable open-loop equilibrium (no controller).
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    T hover_setpoint = params.dynamics.hovering_throttle_relative * 2 - 1;
    const_action<ENV>(action, hover_setpoint);

    for (TI step = 0; step < 20; step++){
        rlt::step(device, env, params, state, action, next_state, rng);
        state = next_state;
    }

    auto& a_state = accel_layer<ENV>(state);
    TI latest = a_state.acceleration_history_step == 0 ? ACCEL_HISTORY_LENGTH - 1 : a_state.acceleration_history_step - 1;
    T body_x = a_state.linear_acceleration_body_history[latest][0];
    T body_y = a_state.linear_acceleration_body_history[latest][1];
    T body_z = a_state.linear_acceleration_body_history[latest][2];
    T expected_g_magnitude = -params.dynamics.gravity[2];
    EXPECT_GT(body_z, 0.5 * expected_g_magnitude)
        << "body z accel " << body_z << " vs expected ~" << expected_g_magnitude;
    EXPECT_GT(std::abs(body_z), 5 * std::abs(body_x))
        << "body_z " << body_z << " body_x " << body_x;
    EXPECT_GT(std::abs(body_z), 5 * std::abs(body_y))
        << "body_z " << body_z << " body_y " << body_y;
    rlt::free(device, action);
}

TEST(L2F_LINEAR_ACCEL_HISTORY, ROTATED_ORIENTATION_MATCHES_ROTATED_WORLD_ACCEL){
    // For an arbitrary orientation, the stored body-frame entry must be exactly
    // R(q')^T * (a_world - g) computed from the integrator's velocity delta.
    // This pins down the post_integration math regardless of physics.
    DEVICE device; RNG rng;
    rlt::init(device); rlt::malloc(device, rng); rlt::init(device, rng, (TI)0);
    ENV env; ENV::Parameters params; ENV::State state, next_state;
    rlt::malloc(device, env); rlt::init(device, env);
    rlt::sample_initial_parameters(device, env, params, rng);
    rlt::initial_state(device, env, params, state);

    T half = M_PI / 4;
    state.orientation[0] = std::cos(half);
    state.orientation[1] = 0;
    state.orientation[2] = std::sin(half);
    state.orientation[3] = 0;
    for (TI d = 0; d < 3; d++) state.linear_velocity[d] = 0.3 * (T)(d + 1);
    for (TI d = 0; d < 3; d++) state.angular_velocity[d] = 0;

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM>> action;
    rlt::malloc(device, action);
    const_action<ENV>(action, 0.6);
    rlt::step(device, env, params, state, action, next_state, rng);

    auto& a_next = accel_layer<ENV>(next_state);
    T body[3] = {a_next.linear_acceleration_body_history[0][0], a_next.linear_acceleration_body_history[0][1], a_next.linear_acceleration_body_history[0][2]};

    T q_conj[4] = {next_state.orientation[0], -next_state.orientation[1], -next_state.orientation[2], -next_state.orientation[3]};
    T a_world[3];
    for (TI d = 0; d < 3; d++)
        a_world[d] = (next_state.linear_velocity[d] - state.linear_velocity[d])/params.integration.dt - params.dynamics.gravity[d];
    T a_body_expected[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(q_conj, a_world, a_body_expected);
    for (TI d = 0; d < 3; d++) EXPECT_NEAR(body[d], a_body_expected[d], 1e-9);
    rlt::free(device, action);
}


TEST(L2F_LINEAR_ACCEL_HISTORY, OBSERVATION_STRING_INCLUDES_HISTORY_LENGTH){
    DEVICE device;
    ENV env; rlt::malloc(device, env); rlt::init(device, env);
    auto s = rlt::string(device, env, ENV::Observation{});
    EXPECT_NE(s.find("LinearAccelerationBodyFrameHistory(" + std::to_string(ACCEL_HISTORY_LENGTH) + ")"), std::string::npos)
        << "got: " << s;
}
