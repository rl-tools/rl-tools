#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/l2f_visual/operations_cpu.h>

#include <gtest/gtest.h>

#include <cmath>

#include "../../../utils/utils.h"

#ifdef RL_TOOLS_TEST_DATA_PATH
static const char* DEFAULT_SCENE_PATH = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH) "/ProcTHOR-Train-1.glb";
#else
static const char* DEFAULT_SCENE_PATH = nullptr;
#endif

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
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
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
}

constexpr TI NUM_ENVS = 4;
constexpr TI CAM_WIDTH = 32;
constexpr TI CAM_HEIGHT = 32;
constexpr TI NUM_PROBES = 8;

using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

using VISUAL_SPEC = rlt::rl::environments::l2f_visual::Specification<T, TI, test_l2f_visual::STATIC_PARAMETERS, NUM_ENVS, CAM_WIDTH, CAM_HEIGHT, NUM_PROBES>;
using ENV = rlt::rl::environments::l2f_visual::MultirrotorVisual<VISUAL_SPEC>;

T dot3(const T a[3], const T b[3]){
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

T norm3(const T v[3]){
    return std::sqrt(dot3(v, v));
}


// user-side composition: a shared library + one renderer + the env-side scene metadata
struct TestVisuals {
    rlt::rendering::raytracing::AssetLibrary<typename ENV::SPEC::RENDERER_SPEC> library;
    rlt::rendering::raytracing::Renderer<typename ENV::SPEC::RENDERER_SPEC> renderer;
    rlt::rendering::raytracing::scene::procthor::Scene<typename ENV::SPEC::SCENE_SPEC> scene;
};

static TestVisuals* setup_visuals(DEVICE& device, ENV& env){
    if(DEFAULT_SCENE_PATH == nullptr){
        return nullptr;
    }
    auto* visuals = new TestVisuals{};
    rlt::malloc(device, visuals->library);
    rlt::malloc(device, visuals->renderer, visuals->library);
    rlt::init(device, visuals->renderer, visuals->library, DEFAULT_SCENE_PATH);
    const T fov = typename ENV::Parameters{}.fov;
    const T up[3] = {0, 0, 1};
    rlt::generate_cameras(device, visuals->renderer, visuals->renderer.scene_center, visuals->renderer.camera_radius, up, fov);
    rlt::generate_probe_directions(device, visuals->renderer);
    rlt::rendering::raytracing::scene::procthor::precompute_indoor_positions(device, visuals->scene, visuals->renderer, fov, (T)ENV::SPEC::CAM_WIDTH / (T)ENV::SPEC::CAM_HEIGHT);
    env.renderer = &visuals->renderer;
    env.scene = &visuals->scene;
    return visuals;
}

static void teardown_visuals(DEVICE& device, TestVisuals* visuals){
    if(visuals != nullptr){
        rlt::free(device, visuals->renderer);
        rlt::free(device, visuals->library);
        delete visuals;
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, LIFECYCLE) {
    DEVICE device;
    ENV env;

    rlt::malloc(device, env);
    auto* visuals = setup_visuals(device, env);
    if(visuals != nullptr){
        EXPECT_NE(env.renderer, nullptr);
        EXPECT_NE(env.scene, nullptr);
        EXPECT_GT(env.scene->num_indoor_positions, 0);
    }

    rlt::init(device, env);

    rlt::free(device, env);
    EXPECT_EQ(env.renderer, nullptr);
    EXPECT_EQ(env.scene, nullptr);
    teardown_visuals(device, visuals);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, SAMPLE_INITIAL_STATE) {
    DEVICE device;
    ENV env;

    rlt::malloc(device, env);
    auto* visuals = setup_visuals(device, env);
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
    teardown_visuals(device, visuals);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, STEP_AND_REWARD) {
    DEVICE device;
    ENV env;

    rlt::malloc(device, env);
    auto* visuals = setup_visuals(device, env);
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
    teardown_visuals(device, visuals);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, OBSERVE_IMAGE) {
    DEVICE device;
    ENV env;

    rlt::malloc(device, env);
    auto* visuals = setup_visuals(device, env);
    rlt::init(device, env);

    if(visuals == nullptr){
        GTEST_SKIP() << "No scene file available, skipping image observation test";
    }

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
    teardown_visuals(device, visuals);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, SAMPLE_INITIAL_PARAMETERS_INITIALIZES_VISUAL_FIELDS) {
    DEVICE device;
    ENV env;
    rlt::init(device, env.dynamics);

    env.parameters.scene_translation[0] = (T)1;
    env.parameters.scene_translation[1] = (T)2;
    env.parameters.scene_translation[2] = (T)3;
    env.parameters.scene_yaw = (T)0.25;
    env.parameters.scene_hash.hash[0] = 0xab;
    env.parameters.camera_mount.offset_body[0] = (T)0.1;
    env.parameters.camera_mount.offset_body[1] = (T)0.2;
    env.parameters.camera_mount.offset_body[2] = (T)0.3;
    env.parameters.camera_mount.forward_body[0] = (T)0;
    env.parameters.camera_mount.forward_body[1] = (T)1;
    env.parameters.camera_mount.forward_body[2] = (T)0;
    env.parameters.fov = (T)1.2;
    env.parameters.camera_randomization.fov_range = (T)0.1;
    env.parameters.collision_distance_threshold = (T)0.33;

    ENV::Parameters parameters;
    parameters.scene_translation[0] = (T)-1;
    parameters.camera_mount.offset_body[0] = (T)-1;
    parameters.fov = (T)-1;

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1);
    rlt::sample_initial_parameters(device, env, parameters, rng);

    EXPECT_FLOAT_EQ(parameters.scene_translation[0], (T)1);
    EXPECT_FLOAT_EQ(parameters.scene_translation[1], (T)2);
    EXPECT_FLOAT_EQ(parameters.scene_translation[2], (T)3);
    EXPECT_FLOAT_EQ(parameters.scene_yaw, (T)0.25);
    EXPECT_EQ(parameters.scene_hash.hash[0], 0xab);
    EXPECT_FLOAT_EQ(parameters.camera_mount.offset_body[0], (T)0.1);
    EXPECT_FLOAT_EQ(parameters.camera_mount.offset_body[1], (T)0.2);
    EXPECT_FLOAT_EQ(parameters.camera_mount.offset_body[2], (T)0.3);
    EXPECT_FLOAT_EQ(parameters.camera_mount.forward_body[0], (T)0);
    EXPECT_FLOAT_EQ(parameters.camera_mount.forward_body[1], (T)1);
    EXPECT_FLOAT_EQ(parameters.camera_mount.forward_body[2], (T)0);
    EXPECT_FLOAT_EQ(parameters.camera_randomization.fov_range, (T)0.1);
    for(TI axis_i = 0; axis_i < 3; axis_i++){
        EXPECT_FLOAT_EQ(parameters.camera_randomization.offset_body_range[axis_i], (T)0);
        EXPECT_FLOAT_EQ(parameters.camera_randomization.rotation_body_range[axis_i], (T)0);
    }
    EXPECT_FLOAT_EQ(parameters.collision_distance_threshold, (T)0.33);
    EXPECT_GE(parameters.fov, (T)1.1);
    EXPECT_LE(parameters.fov, (T)1.3);

    rlt::free(device, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F_VISUAL, CAMERA_MOUNT_RANDOMIZATION_IS_BOUNDED_AND_ORTHONORMAL) {
    DEVICE device;
    ENV env;
    rlt::init(device, env.dynamics);

    env.parameters.camera_mount.offset_body[0] = (T)0.1;
    env.parameters.camera_mount.offset_body[1] = (T)-0.2;
    env.parameters.camera_mount.offset_body[2] = (T)0.3;
    env.parameters.camera_mount.forward_body[0] = (T)1;
    env.parameters.camera_mount.forward_body[1] = (T)0;
    env.parameters.camera_mount.forward_body[2] = (T)0;
    env.parameters.camera_mount.up_body[0] = (T)0;
    env.parameters.camera_mount.up_body[1] = (T)0;
    env.parameters.camera_mount.up_body[2] = (T)1;
    env.parameters.camera_randomization.fov_range = (T)0;
    env.parameters.camera_randomization.offset_body_range[0] = (T)0.01;
    env.parameters.camera_randomization.offset_body_range[1] = (T)0.02;
    env.parameters.camera_randomization.offset_body_range[2] = (T)0.03;
    env.parameters.camera_randomization.rotation_body_range[2] = (T)0.2;

    ENV::Parameters parameters;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 2);
    rlt::sample_initial_parameters(device, env, parameters, rng);

    for(TI axis_i = 0; axis_i < 3; axis_i++){
        T delta = parameters.camera_mount.offset_body[axis_i] - env.parameters.camera_mount.offset_body[axis_i];
        T range = env.parameters.camera_randomization.offset_body_range[axis_i];
        EXPECT_GE(delta, -range - (T)1e-6);
        EXPECT_LE(delta, range + (T)1e-6);
    }

    EXPECT_NEAR(norm3(parameters.camera_mount.forward_body), (T)1, (T)1e-5);
    EXPECT_NEAR(norm3(parameters.camera_mount.up_body), (T)1, (T)1e-5);
    EXPECT_NEAR(dot3(parameters.camera_mount.forward_body, parameters.camera_mount.up_body), (T)0, (T)1e-5);
    EXPECT_NEAR(parameters.camera_mount.forward_body[2], (T)0, (T)1e-5);
    EXPECT_NEAR(parameters.camera_mount.up_body[0], (T)0, (T)1e-5);
    EXPECT_NEAR(parameters.camera_mount.up_body[1], (T)0, (T)1e-5);
    EXPECT_NEAR(parameters.camera_mount.up_body[2], (T)1, (T)1e-5);
    EXPECT_GE(parameters.camera_mount.forward_body[0], std::cos(env.parameters.camera_randomization.rotation_body_range[2]) - (T)1e-5);

    rlt::free(device, rng);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
