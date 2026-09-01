#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/metrics.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/baseline.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
namespace task = rlt::rl::environments::hyperdrone::tasks::visual_inertial_localization;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_visual_inertial_localization {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 2000;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersIMU<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>>>;
    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_IMU = l2f::StateIMU<T, TI, STATE_BASE>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_IMU>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.005}; // 200 Hz IMU rate
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::IMU imu = {{{{0, 0}, {0, 0, 0}}}, {{{0, 0}, {0, 0, 0}}}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{{dynamics, integration, mdp}, imu}, disturbances};
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
    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using BASE_WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    using TASK_SPEC = task::Specification<BASE_WORLD>;
    using WORLD = task::World<TASK_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI OBS_DIM = BASE_WORLD::OBSERVATION_DIM;
    constexpr TI FRAME_STRIDE = WORLD::FRAME_STRIDE;
}

using namespace test_visual_inertial_localization;

struct Tensors {
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, OBS_DIM>>> observations;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ObservationIMU::DIM>>> observations_imu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ObservationGroundTruthPose::DIM>>> observations_pose;
    void allocate(DEVICE& device){
        rlt::malloc(device, parameters);
        rlt::malloc(device, states);
        rlt::malloc(device, next_states);
        rlt::malloc(device, reset_mask);
        rlt::malloc(device, actions);
        rlt::malloc(device, observations);
        rlt::malloc(device, observations_imu);
        rlt::malloc(device, observations_pose);
    }
    void deallocate(DEVICE& device){
        rlt::free(device, parameters);
        rlt::free(device, states);
        rlt::free(device, next_states);
        rlt::free(device, reset_mask);
        rlt::free(device, actions);
        rlt::free(device, observations);
        rlt::free(device, observations_imu);
        rlt::free(device, observations_pose);
    }
};

struct Fixture {
    DEVICE device;
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    RNG rng;
    Tensors tensors;
    void setup(TI seed){
        rlt::init(device);
        rlt::malloc(device, shared.library);
        rlt::malloc(device, world);
        rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
        typename rlt::rendering::datasets::procthor::GLB::Corpus corpus;
        rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
        rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);
        rlt::malloc(device, rng);
        rlt::init(device, rng, seed);
        tensors.allocate(device);
        rlt::set_all(device, tensors.reset_mask, true);
        rlt::set_all(device, tensors.actions, (T)0);
        rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
        rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    }
    void teardown(){
        tensors.deallocate(device);
        rlt::free(device, world);
        rlt::free(device, shared.library);
        rlt::free(device, rng);
    }
};

static void set_hover_actions(DEVICE& device, Tensors& tensors){
    typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
    T hover = instance_parameters.dynamics.dynamics.hovering_throttle_relative * 2 - 1;
    rlt::set_all(device, tensors.actions, hover);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, FRAME_STRIDE_SEMANTICS){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Fixture fixture;
    fixture.setup(1337);
    DEVICE& device = fixture.device;
    WORLD& world = fixture.world;
    Tensors& tensors = fixture.tensors;
    RNG& rng = fixture.rng;
    set_hover_actions(device, tensors);
    {
        // give the drone motion so consecutive frames differ
        typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
        state.linear_velocity[0] = 1;
        rlt::set(device, tensors.states, state, (TI)0);
    }
    std::vector<T> previous_frame(INSTANCES * OBS_DIM);
    for(TI step_i = 0; step_i < 2 * FRAME_STRIDE + 1; step_i++){
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::observe(device, world, tensors.parameters, tensors.states, typename BASE_WORLD::Observation{}, tensors.observations, rng);
        rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::ObservationIMU{}, tensors.observations_imu, rng);
        TI frame_age = step_i % FRAME_STRIDE;
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            EXPECT_NEAR(rlt::get(device, tensors.observations_imu, instance_i, 6), (T)frame_age / (T)FRAME_STRIDE, 1e-6) << "step " << step_i;
            EXPECT_NEAR(rlt::get(device, tensors.observations_imu, instance_i, 7), frame_age == 0 ? (T)1 : (T)0, 1e-6) << "step " << step_i;
        }
        bool frame_changed = false;
        for(TI dim_i = 0; dim_i < INSTANCES * OBS_DIM; dim_i++){
            frame_changed = frame_changed || rlt::get(device, tensors.observations, dim_i / OBS_DIM, dim_i % OBS_DIM) != previous_frame[dim_i];
            previous_frame[dim_i] = rlt::get(device, tensors.observations, dim_i / OBS_DIM, dim_i % OBS_DIM);
        }
        if(step_i == 0){
            EXPECT_TRUE(frame_changed); // first frame vs zero-initialized baseline
        } else if(frame_age == 0){
            EXPECT_TRUE(frame_changed) << "step " << step_i << ": a frame boundary must refresh the frame";
        } else {
            EXPECT_FALSE(frame_changed) << "step " << step_i << ": the frame must be held between boundaries";
        }
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::set_all(device, tensors.reset_mask, false);
    }
    fixture.teardown();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, RESET_REANCHORS_FRAME_PHASE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Fixture fixture;
    fixture.setup(1338);
    DEVICE& device = fixture.device;
    WORLD& world = fixture.world;
    Tensors& tensors = fixture.tensors;
    RNG& rng = fixture.rng;
    set_hover_actions(device, tensors);
    // advance to a mid-stride phase
    for(TI step_i = 0; step_i < FRAME_STRIDE / 2 + 1; step_i++){
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::copy(device, device, tensors.next_states, tensors.states);
        rlt::set_all(device, tensors.reset_mask, false);
    }
    // synchronized reset: fresh episode, fresh frame, phase re-anchored
    rlt::set_all(device, tensors.reset_mask, true);
    rlt::sample_initial_parameters(device, world, tensors.parameters, tensors.reset_mask, rng);
    rlt::sample_initial_state(device, world, tensors.parameters, tensors.states, tensors.reset_mask, rng);
    rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::ObservationIMU{}, tensors.observations_imu, rng);
    EXPECT_NEAR(rlt::get(device, tensors.observations_imu, (TI)0, 6), (T)0, 1e-6);
    EXPECT_NEAR(rlt::get(device, tensors.observations_imu, (TI)0, 7), (T)1, 1e-6);
    fixture.teardown();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, ROUTE_AND_WAYPOINT_PROGRESSION){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Fixture fixture;
    fixture.setup(1339);
    DEVICE& device = fixture.device;
    WORLD& world = fixture.world;
    Tensors& tensors = fixture.tensors;
    RNG& rng = fixture.rng;
    typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
    typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        EXPECT_EQ(instance_parameters.waypoints[0][dim_i], (T)0) << "waypoint 0 anchors at the dynamics origin";
    }
    EXPECT_EQ(state.current_waypoint, 1);
    for(TI waypoint_i = 1; waypoint_i < WORLD::Parameters::NUM_WAYPOINTS; waypoint_i++){
        T distance_squared = 0;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            T value = instance_parameters.waypoints[waypoint_i][dim_i];
            EXPECT_LT(std::abs(value), 1000);
            T delta = value - instance_parameters.waypoints[waypoint_i - 1][dim_i];
            distance_squared += delta * delta;
        }
        EXPECT_GT(distance_squared, 0) << "consecutive waypoints must be distinct";
    }
    // place the drone next to its current waypoint: the step should advance the index
    state.position[0] = instance_parameters.waypoints[1][0];
    state.position[1] = instance_parameters.waypoints[1][1];
    state.position[2] = instance_parameters.waypoints[1][2];
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        state.linear_velocity[dim_i] = 0;
        state.angular_velocity[dim_i] = 0;
    }
    rlt::set(device, tensors.states, state, (TI)0);
    set_hover_actions(device, tensors);
    rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
    typename WORLD::State next_state = rlt::get(device, tensors.next_states, (TI)0);
    EXPECT_EQ(next_state.current_waypoint, 2);
    typename WORLD::State next_state_1 = rlt::get(device, tensors.next_states, (TI)1);
    EXPECT_EQ(next_state_1.current_waypoint, 1) << "instance 1 is far from its waypoint and must not advance";
    fixture.teardown();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, GROUND_TRUTH_POSE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Fixture fixture;
    fixture.setup(1340);
    DEVICE& device = fixture.device;
    WORLD& world = fixture.world;
    Tensors& tensors = fixture.tensors;
    RNG& rng = fixture.rng;
    typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
    typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
    state.position[0] = (T)0.3;
    state.position[1] = (T)-0.2;
    state.position[2] = (T)0.1;
    rlt::set(device, tensors.states, state, (TI)0);
    rlt::observe(device, world, tensors.parameters, tensors.states, typename WORLD::ObservationGroundTruthPose{}, tensors.observations_pose, rng);
    T expected_x = instance_parameters.scene_yaw_cos * state.position[0] - instance_parameters.scene_yaw_sin * state.position[1] + instance_parameters.scene_translation[0];
    T expected_y = instance_parameters.scene_yaw_sin * state.position[0] + instance_parameters.scene_yaw_cos * state.position[1] + instance_parameters.scene_translation[1];
    T expected_z = state.position[2] + instance_parameters.scene_translation[2];
    EXPECT_NEAR(rlt::get(device, tensors.observations_pose, (TI)0, 0), expected_x, 1e-5);
    EXPECT_NEAR(rlt::get(device, tensors.observations_pose, (TI)0, 1), expected_y, 1e-5);
    EXPECT_NEAR(rlt::get(device, tensors.observations_pose, (TI)0, 2), expected_z, 1e-5);
    // rotating the body x axis by the observed quaternion must match yaw-rotating the
    // state-quaternion-rotated body x axis
    T observed_orientation[4];
    for(TI dim_i = 0; dim_i < 4; dim_i++){
        observed_orientation[dim_i] = rlt::get(device, tensors.observations_pose, (TI)0, 3 + dim_i);
    }
    T body_x[3] = {1, 0, 0};
    T forward_dynamics[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(state.orientation, body_x, forward_dynamics);
    T forward_scene_expected[3];
    forward_scene_expected[0] = instance_parameters.scene_yaw_cos * forward_dynamics[0] - instance_parameters.scene_yaw_sin * forward_dynamics[1];
    forward_scene_expected[1] = instance_parameters.scene_yaw_sin * forward_dynamics[0] + instance_parameters.scene_yaw_cos * forward_dynamics[1];
    forward_scene_expected[2] = forward_dynamics[2];
    T forward_scene_observed[3];
    l2f::rotate_vector_by_quaternion<DEVICE, T>(observed_orientation, body_x, forward_scene_observed);
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        EXPECT_NEAR(forward_scene_observed[dim_i], forward_scene_expected[dim_i], 1e-5);
    }
    fixture.teardown();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, DEAD_RECKONING_ZERO_NOISE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Fixture fixture;
    fixture.setup(1341);
    DEVICE& device = fixture.device;
    WORLD& world = fixture.world;
    Tensors& tensors = fixture.tensors;
    RNG& rng = fixture.rng;
    // the preset-independent test config has zero IMU noise/bias by default; verify
    typename WORLD::Parameters instance_parameters = rlt::get(device, tensors.parameters, (TI)0);
    ASSERT_EQ(instance_parameters.dynamics.imu.accelerometer.error.noise.std, (T)0);
    ASSERT_EQ(instance_parameters.dynamics.imu.gyro.error.noise.std, (T)0);
    const T dt = instance_parameters.dynamics.integration.dt;
    const T* gravity = instance_parameters.dynamics.dynamics.gravity;
    typename WORLD::State state = rlt::get(device, tensors.states, (TI)0);
    task::DeadReckoningState<T> estimator{};
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        estimator.position[dim_i] = state.position[dim_i];
        estimator.linear_velocity[dim_i] = state.linear_velocity[dim_i];
    }
    for(TI dim_i = 0; dim_i < 4; dim_i++){
        estimator.orientation[dim_i] = state.orientation[dim_i];
    }
    task::DeadReckoningState<T> oracle_estimator = estimator; // fed ground-truth orientation each step
    set_hover_actions(device, tensors);
    constexpr TI N_STEPS = 100; // 0.5 s at 200 Hz
    for(TI step_i = 0; step_i < N_STEPS; step_i++){
        rlt::render(device, world, tensors.parameters, tensors.states, tensors.reset_mask);
        rlt::set_all(device, tensors.reset_mask, false);
        // small excitation around hover
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++){
                T excitation = rlt::random::uniform_real_distribution(device.random, (T)-0.05, (T)0.05, rng);
                rlt::set(device, tensors.actions, rlt::get(device, tensors.actions, instance_i, action_i) + excitation, instance_i, action_i);
            }
        }
        rlt::step(device, world, tensors.parameters, tensors.states, tensors.actions, tensors.next_states, rng);
        rlt::observe(device, world, tensors.parameters, tensors.next_states, typename WORLD::ObservationIMU{}, tensors.observations_imu, rng);
        T accelerometer[3], gyroscope[3];
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            accelerometer[dim_i] = rlt::get(device, tensors.observations_imu, (TI)0, dim_i);
            gyroscope[dim_i] = rlt::get(device, tensors.observations_imu, (TI)0, 3 + dim_i);
        }
        task::dead_reckoning_step(device, estimator, accelerometer, gyroscope, gravity, dt);
        // oracle attitude: substitute the true orientation, isolating the accelerometer channel
        typename WORLD::State next_state = rlt::get(device, tensors.next_states, (TI)0);
        for(TI dim_i = 0; dim_i < 4; dim_i++){
            oracle_estimator.orientation[dim_i] = next_state.orientation[dim_i];
        }
        T acceleration_world[3];
        l2f::rotate_vector_by_quaternion<DEVICE, T>(oracle_estimator.orientation, accelerometer, acceleration_world);
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            T velocity_next = oracle_estimator.linear_velocity[dim_i] + (acceleration_world[dim_i] + gravity[dim_i]) * dt;
            oracle_estimator.position[dim_i] += (oracle_estimator.linear_velocity[dim_i] + velocity_next) * dt / (T)2;
            oracle_estimator.linear_velocity[dim_i] = velocity_next;
        }
        rlt::copy(device, device, tensors.next_states, tensors.states);
        set_hover_actions(device, tensors);
    }
    typename WORLD::State final_state = rlt::get(device, tensors.states, (TI)0);
    // the oracle (true attitude) recovers the velocity to float round-off
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        EXPECT_NEAR(oracle_estimator.linear_velocity[dim_i], final_state.linear_velocity[dim_i], 1e-3) << "axis " << dim_i;
        EXPECT_NEAR(oracle_estimator.position[dim_i], final_state.position[dim_i], 1e-2) << "axis " << dim_i;
    }
    // full dead reckoning adds the gyro-integrated attitude error
    task::TrajectoryMetricsAccumulator<T, TI> accumulator{};
    T ground_truth_orientation[4], estimated_orientation[4];
    for(TI dim_i = 0; dim_i < 4; dim_i++){
        ground_truth_orientation[dim_i] = final_state.orientation[dim_i];
        estimated_orientation[dim_i] = estimator.orientation[dim_i];
    }
    T rotation_error = task::quaternion_geodesic_distance(device, ground_truth_orientation, estimated_orientation);
    EXPECT_LT(rotation_error, 1e-2);
    for(TI dim_i = 0; dim_i < 3; dim_i++){
        EXPECT_NEAR(estimator.position[dim_i], final_state.position[dim_i], 5e-2) << "axis " << dim_i;
    }
    fixture.teardown();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_VISUAL_INERTIAL_LOCALIZATION, METRICS){
    DEVICE device;
    rlt::init(device);
    task::TrajectoryMetricsAccumulator<T, TI> accumulator{};
    task::reset(device, accumulator);
    T identity[4] = {1, 0, 0, 0};
    T rotated_90_z[4] = {(T)std::sqrt(0.5), 0, 0, (T)std::sqrt(0.5)};
    T origin[3] = {0, 0, 0};
    T offset_34[3] = {3, 4, 0};   // |error| = 5
    T step_position[3] = {1, 0, 0};
    task::accumulate(device, accumulator, origin, identity, offset_34, identity);
    task::accumulate(device, accumulator, step_position, identity, step_position, rotated_90_z);
    EXPECT_NEAR(task::ate_position_rmse(device, accumulator), std::sqrt((T)25 / 2), 1e-5);
    EXPECT_NEAR(task::mean_rotation_error(device, accumulator), (T)(rlt::math::PI<T> / 2) / 2, 1e-5);
    EXPECT_NEAR(task::max_rotation_error(device, accumulator), (T)(rlt::math::PI<T> / 2), 1e-5);
    EXPECT_NEAR(accumulator.distance_traveled, (T)1, 1e-6);
    // the sign-flipped quaternion is the same rotation
    T negated_identity[4] = {-1, 0, 0, 0};
    EXPECT_NEAR(task::quaternion_geodesic_distance(device, identity, negated_identity), (T)0, 1e-6);
    // relative pose: origin at (1,2,3) yawed 90 deg; the pose one local x ahead and yawed
    // another 90 deg must come out as ((1,0,0), 90 deg yaw) in the origin frame
    T origin_position[3] = {1, 2, 3};
    T pose_position[3] = {1, 3, 3};
    T yaw_180_z[4] = {0, 0, 0, 1};
    T relative_position[3], relative_orientation[4];
    task::relative_pose(device, origin_position, rotated_90_z, pose_position, yaw_180_z, relative_position, relative_orientation);
    EXPECT_NEAR(relative_position[0], (T)1, 1e-5);
    EXPECT_NEAR(relative_position[1], (T)0, 1e-5);
    EXPECT_NEAR(relative_position[2], (T)0, 1e-5);
    EXPECT_NEAR(task::quaternion_geodesic_distance(device, relative_orientation, rotated_90_z), (T)0, 1e-3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
