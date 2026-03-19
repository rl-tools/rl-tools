#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;

namespace test_l2f_visual {
    namespace l2f = rlt::rl::environments::l2f;
    namespace obs = l2f::observation;

    using REWARD_FUNCTION = rlt::rl::environments::l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    static constexpr auto MODEL = rlt::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;

    struct STATIC_PARAMETERS {
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;

        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = obs::Position<obs::PositionSpecification<T, TI,
                obs::OrientationRotationMatrix<obs::OrientationRotationMatrixSpecification<T, TI,
                obs::LinearVelocity<obs::LinearVelocitySpecification<T, TI,
                obs::AngularVelocity<obs::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = rlt::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)1/(T)SIMULATION_FREQUENCY};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = rlt::rl::environments::l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
        static constexpr T STATE_LIMIT_POSITION = 100000;
        static constexpr T STATE_LIMIT_VELOCITY = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
    };
}

constexpr TI NUM_ENVS = 4;
constexpr TI CAM_WIDTH = 32;
constexpr TI CAM_HEIGHT = 32;
constexpr TI NUM_PROBES = 8;

using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, test_l2f_visual::STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENV = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

TEST(RlEnvironmentsL2fVisual, Lifecycle) {
    DEVICE device;
    ENV env;
    env.scene_path = nullptr;

    rlt::malloc(device, env);
    EXPECT_NE(env.renderer, nullptr);
    EXPECT_NE(env.scene, nullptr);
    EXPECT_TRUE(env.owns_renderer);

    rlt::init(device, env);
    EXPECT_GT(env.scene->num_indoor_positions, 0);

    rlt::free(device, env);
    EXPECT_EQ(env.renderer, nullptr);
    EXPECT_EQ(env.scene, nullptr);
}

TEST(RlEnvironmentsL2fVisual, SampleInitialState) {
    DEVICE device;
    ENV env;
    env.scene_path = nullptr;

    rlt::malloc(device, env);
    rlt::init(device, env);

    ENV::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);

    RNG rng;
    ENV::State state;
    rlt::sample_initial_state(device, env, parameters, state, rng);

    T qnorm = state.orientation[0]*state.orientation[0] + state.orientation[1]*state.orientation[1]
             + state.orientation[2]*state.orientation[2] + state.orientation[3]*state.orientation[3];
    EXPECT_NEAR(qnorm, 1.0, 1e-5);

    for (TI i = 0; i < 3; i++) {
        EXPECT_FLOAT_EQ(state.linear_velocity[i], 0.0f);
        EXPECT_FLOAT_EQ(state.angular_velocity[i], 0.0f);
    }

    rlt::free(device, env);
}

TEST(RlEnvironmentsL2fVisual, StepAndReward) {
    DEVICE device;
    ENV env;
    env.scene_path = nullptr;

    rlt::malloc(device, env);
    rlt::init(device, env);

    ENV::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);

    RNG rng;
    ENV::State state, next_state;
    rlt::sample_initial_state(device, env, parameters, state, rng);

    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENV::ACTION_DIM, false>> action;
    for (TI i = 0; i < ENV::ACTION_DIM; i++) {
        rlt::set(action, 0, i, static_cast<T>(0));
    }

    T dt = rlt::step(device, env, parameters, state, action, next_state, rng);
    EXPECT_GT(dt, 0);

    T r = rlt::reward(device, env, parameters, state, action, next_state, rng);
    EXPECT_TRUE(std::isfinite(r));

    bool term = rlt::terminated(device, env, parameters, next_state, rng);
    EXPECT_FALSE(term);

    rlt::free(device, env);
}

TEST(RlEnvironmentsL2fVisual, ObserveImage) {
    DEVICE device;
    ENV env;
    env.scene_path = nullptr;

    rlt::malloc(device, env);
    rlt::init(device, env);

    ENV::Parameters parameters;
    rlt::initial_parameters(device, env, parameters);

    RNG rng;
    ENV::State state;
    rlt::sample_initial_state(device, env, parameters, state, rng);

    constexpr TI OBS_DIM = ENV::Observation::DIM;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, OBS_DIM, true>> observation;
    rlt::malloc(device, observation);

    ENV::Observation obs_type;
    rlt::observe(device, env, parameters, state, obs_type, observation, rng);

    T sum = 0;
    for (TI i = 0; i < OBS_DIM; i++) {
        T val = rlt::get(observation, 0, i);
        EXPECT_GE(val, 0.0f);
        EXPECT_LE(val, 1.0f);
        sum += val;
    }
    EXPECT_GT(sum, 0);

    rlt::free(device, observation);
    rlt::free(device, env);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
