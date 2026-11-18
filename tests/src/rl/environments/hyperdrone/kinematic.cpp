#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace hd = rlt::rl::environments::hyperdrone;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_kinematic {
    template <typename T>
    struct Limits {
        static constexpr T LINEAR_VELOCITY = 2;
        static constexpr T YAW_RATE = 1.5;
    };
    template <typename T>
    using ENVIRONMENT = hd::KinematicMultirotor<l2f::Specification<T, TI, hd::presets::x500_fpv::DynamicsStaticParameters<T, TI>>, Limits<T>>;

    struct Pose {
        double position[3];
        double yaw;
    };
    // constant body twist (linear_velocity in the level body frame, yaw_rate) from start over time
    Pose integrate_twist(const Pose& start, const double linear_velocity[3], double yaw_rate, double time) {
        const double turned = yaw_rate * time;
        const double along = yaw_rate == 0 ? time : std::sin(turned) / yaw_rate;
        const double across = yaw_rate == 0 ? 0 : (1 - std::cos(turned)) / yaw_rate;
        const double heading_x = along * linear_velocity[0] - across * linear_velocity[1];
        const double heading_y = across * linear_velocity[0] + along * linear_velocity[1];
        Pose end;
        end.position[0] = start.position[0] + std::cos(start.yaw) * heading_x - std::sin(start.yaw) * heading_y;
        end.position[1] = start.position[1] + std::sin(start.yaw) * heading_x + std::cos(start.yaw) * heading_y;
        end.position[2] = start.position[2] + linear_velocity[2] * time;
        end.yaw = start.yaw + turned;
        return end;
    }
    template <typename STATE>
    double yaw(const STATE& state) {
        return std::atan2(2.0 * state.orientation[0] * state.orientation[3], (double)state.orientation[0] * state.orientation[0] - (double)state.orientation[3] * state.orientation[3]);
    }
    double angle_difference(double a, double b) {
        return std::remainder(a - b, 2 * M_PI);
    }
    template <typename STATE>
    void expect_level(const STATE& state, double tolerance) {
        EXPECT_EQ(state.orientation[1], 0);
        EXPECT_EQ(state.orientation[2], 0);
        EXPECT_NEAR((double)state.orientation[0] * state.orientation[0] + (double)state.orientation[3] * state.orientation[3], 1, tolerance);
    }
    // at rest includes a zero command history (the l2f initial state would seed it from the hover rotor speed)
    template <typename STATE>
    void expect_at_rest(const STATE& state) {
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            EXPECT_EQ(state.linear_velocity[axis_i], 0);
            EXPECT_EQ(state.angular_velocity[axis_i], 0);
        }
        for (const auto value : state.last_action) {
            EXPECT_EQ(value, 0);
        }
        for (const auto& step : state.action_history) {
            for (const auto value : step) {
                EXPECT_EQ(value, 0);
            }
        }
    }
    // expects the state to be the constant-twist pose with the velocity it ends with; returns the largest pose error
    template <typename STATE>
    double expect_twist_pose(const STATE& state, const Pose& expected, const double linear_velocity[3], double yaw_rate, double tolerance) {
        double error = 0;
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            EXPECT_NEAR(state.position[axis_i], expected.position[axis_i], tolerance) << "axis " << axis_i;
            error = std::max(error, std::abs(state.position[axis_i] - expected.position[axis_i]));
        }
        const double yaw_error = angle_difference(yaw(state), expected.yaw);
        EXPECT_NEAR(yaw_error, 0, tolerance);
        error = std::max(error, std::abs(yaw_error));
        expect_level(state, tolerance);
        EXPECT_NEAR(state.linear_velocity[0], std::cos(expected.yaw) * linear_velocity[0] - std::sin(expected.yaw) * linear_velocity[1], tolerance);
        EXPECT_NEAR(state.linear_velocity[1], std::sin(expected.yaw) * linear_velocity[0] + std::cos(expected.yaw) * linear_velocity[1], tolerance);
        EXPECT_NEAR(state.linear_velocity[2], linear_velocity[2], tolerance);
        EXPECT_EQ(state.angular_velocity[0], 0);
        EXPECT_EQ(state.angular_velocity[1], 0);
        EXPECT_NEAR(state.angular_velocity[2], yaw_rate, tolerance);
        return error;
    }

    using T = float;
    struct WORLD_SPEC: hd::Specification<T, TI, hd::presets::x500_fpv::DynamicsStaticParameters<T, TI>> {
        template <typename DYNAMICS_SPEC>
        using DYNAMICS_ENV = hd::KinematicMultirotor<DYNAMICS_SPEC, Limits<T>>;
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        static constexpr TI HISTORY_LENGTH = 2;
        using SHADING = rlt::hyperlux::Low;
    };
    using BASE_WORLD = hd::World<WORLD_SPEC>;
    static_assert(rlt::utils::typing::is_same_v<typename BASE_WORLD::DYNAMICS_ENV, hd::KinematicMultirotor<l2f::Specification<T, TI, typename BASE_WORLD::DYNAMICS_STATIC_PARAMETERS>, Limits<T>>>);
    static_assert(rlt::utils::typing::is_same_v<typename hd::DroneParameters<WORLD_SPEC>::DYNAMICS_ENV, typename BASE_WORLD::DYNAMICS_ENV>);
    static_assert(BASE_WORLD::ACTION_DIM == 4);
    struct TASK_SPEC: hd::tasks::target_frame::Specification<BASE_WORLD> {
        static constexpr TI IMAGE_STACK_N = 2;
        static constexpr TI TARGET_BANK_REFRESH_STEPS = 4;
    };
    using WORLD = hd::tasks::target_frame::World<TASK_SPEC>;
    static constexpr TI INSTANCES = WORLD::INSTANCES;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC, CONSTANT_TWIST_IS_INTEGRATED_EXACTLY_AT_EVERY_RATE) {
    using namespace test_kinematic;
    using T = double;
    using ENV = ENVIRONMENT<T>;
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    ENV env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV::Parameters parameters = env.parameters;
    // yaw rates through zero, the small-angle branch and its threshold; saturated commands clamp
    const T commands[][4] = {{0.5, -0.25, 0.4, 0.8}, {0.5, -0.25, 0.4, 0}, {0.5, -0.25, 0.4, 1e-6}, {0.5, -0.25, 0.4, 0.0133}, {-0.7, 1, -1, -1}, {-0.7, 3, -2, -5}};
    const T dts[] = {0.01, 0.05, 0.25};
    const Pose start = {{0.3, -0.2, 1.0}, 2.9};
    double max_error = 0;
    for (const auto& command : commands) {
        double linear_velocity[3];
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            linear_velocity[axis_i] = std::clamp(command[axis_i], -1.0, 1.0) * Limits<T>::LINEAR_VELOCITY;
        }
        const double yaw_rate = std::clamp(command[3], -1.0, 1.0) * Limits<T>::YAW_RATE;
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>> action;
        for (TI action_i = 0; action_i < 4; action_i++) {
            rlt::set(action, 0, action_i, command[action_i]);
        }
        for (const T dt : dts) {
            parameters.integration.dt = dt;
            typename ENV::State state, next_state;
            rlt::initial_state(device, env, parameters, state);
            for (TI axis_i = 0; axis_i < 3; axis_i++) {
                state.position[axis_i] = start.position[axis_i];
            }
            state.orientation[0] = std::cos(start.yaw / 2);
            state.orientation[3] = std::sin(start.yaw / 2);
            const TI steps = (TI)std::lround(1 / dt);
            for (TI step_i = 1; step_i <= steps; step_i++) {
                EXPECT_EQ(rlt::step(device, env, parameters, state, action, next_state, rng), dt);
                state = next_state;
                max_error = std::max(max_error, expect_twist_pose(state, integrate_twist(start, linear_velocity, yaw_rate, step_i * dt), linear_velocity, yaw_rate, 1e-10));
                for (TI action_i = 0; action_i < 4; action_i++) {
                    EXPECT_EQ(state.last_action[action_i], std::clamp(command[action_i], -1.0, 1.0));
                }
            }
        }
    }
    metra::log("hyperdrone/kinematic/closed_form_max_error", max_error);
    rlt::free(device, env);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC, RESET_IS_LEVEL_AT_REST_WITHIN_THE_INITIALIZATION_RANGE) {
    using namespace test_kinematic;
    using T = double;
    using ENV = ENVIRONMENT<T>;
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    ENV env;
    rlt::malloc(device, env);
    rlt::init(device, env);
    typename ENV::Parameters parameters = env.parameters;
    auto& init = parameters.mdp.init;
    const double centre_yaw = 0.3;
    const double centre_roll = 0.4;
    // R_z(0.3) * R_x(0.4): the tilted centre's twist about z is R_z(0.3)
    init.orientation[0] = std::cos(centre_yaw / 2) * std::cos(centre_roll / 2);
    init.orientation[1] = std::cos(centre_yaw / 2) * std::sin(centre_roll / 2);
    init.orientation[2] = std::sin(centre_yaw / 2) * std::sin(centre_roll / 2);
    init.orientation[3] = std::sin(centre_yaw / 2) * std::cos(centre_roll / 2);
    const T centre[3] = {1, 2, 3};
    for (TI axis_i = 0; axis_i < 3; axis_i++) {
        init.position[axis_i] = centre[axis_i];
    }
    init.guidance = 0;
    init.max_position = 0.5;
    init.max_angle = 0.7;
    typename ENV::State state;
    double max_position_offset = 0;
    double max_yaw_offset = 0;
    for (TI sample_i = 0; sample_i < 4000; sample_i++) {
        rlt::sample_initial_state(device, env, parameters, state, rng);
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            const double offset = std::abs(state.position[axis_i] - centre[axis_i]);
            EXPECT_LE(offset, init.max_position + 1e-12);
            max_position_offset = std::max(max_position_offset, offset);
        }
        const double yaw_offset = std::abs(angle_difference(yaw(state), centre_yaw));
        EXPECT_LE(yaw_offset, init.max_angle + 1e-12);
        max_yaw_offset = std::max(max_yaw_offset, yaw_offset);
        expect_level(state, 1e-12);
        expect_at_rest(state);
    }
    EXPECT_GT(max_position_offset, 0.49);
    EXPECT_GT(max_yaw_offset, 0.69);
    init.guidance = 1;
    rlt::sample_initial_state(device, env, parameters, state, rng);
    for (TI axis_i = 0; axis_i < 3; axis_i++) {
        EXPECT_EQ(state.position[axis_i], centre[axis_i]);
    }
    EXPECT_NEAR(angle_difference(yaw(state), centre_yaw), 0, 1e-12);
    expect_level(state, 1e-12);
    expect_at_rest(state);
    rlt::free(device, env);
    rlt::free(device, rng);
}

// the World's per-instance verbs bind the kinematic overloads: the x500 initialization (90 deg, 1 m/s) would tilt and move an l2f start, and a full motor command would barely displace it in one step
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC, WORLD_ROUTES_ITS_DYNAMICS_THROUGH_THE_KINEMATIC_MODEL) {
    using namespace test_kinematic;
    DEVICE device;
    rlt::init(device);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    BASE_WORLD world;
    typename BASE_WORLD::Parameters parameters = world.parameters;
    typename BASE_WORLD::State state, next_state;
    for (TI sample_i = 0; sample_i < 100; sample_i++) {
        hd::_sample_dynamics_state<DEVICE, WORLD_SPEC>(device, world.dynamics, parameters, state, rng);
        expect_level(state, 1e-6);
        expect_at_rest(state);
    }
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>> action;
    const T command[4] = {1, 0, 0, 0};
    for (TI action_i = 0; action_i < 4; action_i++) {
        rlt::set(action, 0, action_i, command[action_i]);
    }
    const T dt = hd::_step<DEVICE, WORLD_SPEC>(device, world.dynamics, parameters, state, action, next_state, rng);
    const double heading = yaw(state);
    EXPECT_NEAR(next_state.position[0] - state.position[0], std::cos(heading) * Limits<T>::LINEAR_VELOCITY * dt, 1e-6);
    EXPECT_NEAR(next_state.position[1] - state.position[1], std::sin(heading) * Limits<T>::LINEAR_VELOCITY * dt, 1e-6);
    EXPECT_NEAR(next_state.position[2] - state.position[2], 0, 1e-6);
    rlt::free(device, rng);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC, TARGET_FRAME_EPISODE_FOLLOWS_THE_COMMANDED_TWIST) {
    using namespace test_kinematic;
    if (SCENE_PATH.empty()) {
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    WORLD world;
    typename BASE_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    rlt::hyperlux::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::hyperlux::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>> states, next_states;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, INSTANCES>>> reset_mask;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>> actions;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::OBSERVATION_DIM>>> observations;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, next_states);
    rlt::malloc(device, reset_mask);
    rlt::malloc(device, actions);
    rlt::malloc(device, observations);

    rlt::set_all(device, reset_mask, true);
    rlt::sample_initial_parameters(device, world, parameters, reset_mask, rng);
    rlt::sample_initial_state(device, world, parameters, states, reset_mask, rng);
    rlt::observe(device, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    std::vector<T> first_observations(INSTANCES * WORLD::OBSERVATION_DIM);
    for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
        for (TI element_i = 0; element_i < WORLD::OBSERVATION_DIM; element_i++) {
            first_observations[instance_i * WORLD::OBSERVATION_DIM + element_i] = rlt::get(device, observations, instance_i, element_i);
        }
    }
    const T commands[INSTANCES][4] = {{0.5, 0, 0, 0.5}, {0, -0.5, 0.25, -1}};
    std::vector<Pose> starts(INSTANCES);
    for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
        const auto& state = rlt::get_ref(device, states, instance_i);
        const auto& init = rlt::get_ref(device, parameters, instance_i).agents[0].dynamics.mdp.init;
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            EXPECT_LE(std::abs(state.position[axis_i] - init.position[axis_i]), init.max_position);
            starts[instance_i].position[axis_i] = state.position[axis_i];
        }
        starts[instance_i].yaw = yaw(state);
        EXPECT_LE(std::abs(starts[instance_i].yaw), init.max_angle + 1e-5);
        expect_level(state, 1e-6);
        expect_at_rest(state);
        for (TI action_i = 0; action_i < 4; action_i++) {
            rlt::set(device, actions, commands[instance_i][action_i], instance_i, action_i);
        }
    }

    constexpr TI STEPS = 10;
    rlt::set_all(device, reset_mask, false);
    for (TI step_i = 0; step_i < STEPS; step_i++) {
        rlt::step(device, world, parameters, states, actions, next_states, rng);
        rlt::copy(device, device, next_states, states);
    }
    rlt::observe(device, world, parameters, states, typename WORLD::Observation{}, observations, rng);
    for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
        const T dt = rlt::get_ref(device, parameters, instance_i).agents[0].dynamics.integration.dt;
        double linear_velocity[3];
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            linear_velocity[axis_i] = commands[instance_i][axis_i] * Limits<T>::LINEAR_VELOCITY;
        }
        const double yaw_rate = commands[instance_i][3] * Limits<T>::YAW_RATE;
        expect_twist_pose(rlt::get_ref(device, states, instance_i), integrate_twist(starts[instance_i], linear_velocity, yaw_rate, STEPS * dt), linear_velocity, yaw_rate, 1e-5);
        bool stack_changed = false;
        for (TI pixel_i = 0; pixel_i < WORLD_SPEC::CAM_WIDTH * WORLD_SPEC::CAM_HEIGHT; pixel_i++) {
            for (TI channel_i = 0; channel_i < WORLD::OBSERVATION_CHANNELS; channel_i++) {
                const TI element_i = pixel_i * WORLD::OBSERVATION_CHANNELS + channel_i;
                const T value = rlt::get(device, observations, instance_i, element_i);
                ASSERT_TRUE(value >= 0 && value <= 1) << "instance " << instance_i << " element " << element_i;
                const T first = first_observations[instance_i * WORLD::OBSERVATION_DIM + element_i];
                if (channel_i < BASE_WORLD::IMAGE_CHANNELS) {
                    stack_changed = stack_changed || value != first;
                }
                if (channel_i >= TASK_SPEC::IMAGE_STACK_N * BASE_WORLD::IMAGE_CHANNELS) {
                    ASSERT_EQ(value, first) << "the target frame must stay fixed within an episode";
                }
            }
        }
        EXPECT_TRUE(stack_changed) << "instance " << instance_i << ": the newest frame should follow the drone";
    }

    rlt::free(device, parameters);
    rlt::free(device, states);
    rlt::free(device, next_states);
    rlt::free(device, reset_mask);
    rlt::free(device, actions);
    rlt::free(device, observations);
    rlt::free(device, rng);
    rlt::free(device, world);
    rlt::free(device, shared.library);
}
