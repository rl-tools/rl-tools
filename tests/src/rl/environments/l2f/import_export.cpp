#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

#include <gtest/gtest.h>

#include "../../../utils/utils.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;


constexpr T EPSILON = 1e-6;

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI>;
using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, IMPORT_EXPORT){
    DEVICE device;
    RNG rng;
    rlt::init(device);
    rlt::malloc(device, rng);
    TI seed = 0;
    rlt::init(device, rng, seed);

    ENVIRONMENT env;
    ENVIRONMENT::Parameters params, params_reconstruct;
    params = {};
    ENVIRONMENT::State state, state_reconstruct;

    rlt::malloc(device, env);
    env.parameters.dynamics.rotor_positions[0][0] = 1337;
    rlt::init(device, env);
    env.parameters.trajectory.steps[7].position[1] = 1337;
    env.parameters.trajectory_parameters.type = rlt::rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS;
    env.parameters.trajectory_parameters.parameters.lissajous = rlt::rl::environments::l2f::parameters::trajectories::lissajous::default_parameters<T>;
    env.parameters.trajectory_parameters.parameters.lissajous.b = 1337;
    env.parameters.observation_delay.linear_velocity = 1337;
    env.parameters.observation_delay.angular_velocity = 1338;
    unsigned char* raw = reinterpret_cast<unsigned char*>(&params);
    unsigned char* raw_reconstruct = reinterpret_cast<unsigned char*>(&params_reconstruct);
    std::generate(raw, raw + sizeof(params), [] { return static_cast<unsigned char>(std::rand() % 256); });
    std::generate(raw_reconstruct, raw_reconstruct + sizeof(params_reconstruct), [] { return static_cast<unsigned char>(std::rand() % 256); });
    rlt::sample_initial_parameters(device, env, params, rng);
    params.trajectory.steps[8].position[1] = 1337;
    ASSERT_NE(params.trajectory.steps[7].position[1], 1337);
    // trajectory_parameters.type is always LISSAJOUS since it's the only type
    ASSERT_EQ(params.trajectory_parameters.type, rlt::rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS);
    ASSERT_EQ(params.trajectory_parameters.parameters.lissajous.b, 1337);
    ASSERT_NE(params.dynamics.rotor_positions[0][0], 1337);
    rlt::sample_initial_state(device, env, params, state, rng);
    auto params_json = rlt::json(device, env, params);
    // Set valid type before exporting params_reconstruct (which was filled with random bytes)
    params_reconstruct.trajectory_parameters.type = rlt::rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS;
    auto params_json_reconstruct = rlt::json(device, env, params_reconstruct);
    ASSERT_NE(params_json_reconstruct, params_json);

    nlohmann::json params_json_object = nlohmann::json::parse(params_json);
    rlt::from_json(device, env, params_json, params_reconstruct);
    static constexpr T EPSILON = 1e-5;
    ASSERT_NE(params_json.find("\"imu\""), std::string::npos);
    ASSERT_EQ(rlt::abs_diff(device, params, params_reconstruct), 0);

    params_json_reconstruct = rlt::json(device, env, params_reconstruct);
    ASSERT_EQ(params_json_reconstruct, params_json);
    ASSERT_NEAR(params_reconstruct.observation_delay.linear_velocity, 1337, EPSILON);
    ASSERT_NEAR(params_reconstruct.observation_delay.angular_velocity, 1338, EPSILON);

    ASSERT_NEAR(params_reconstruct.trajectory.steps[8].position[1], 1337, EPSILON);
    ASSERT_EQ(params_reconstruct.trajectory_parameters.type, rlt::rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS);
    ASSERT_NEAR(params_reconstruct.trajectory_parameters.parameters.lissajous.b, 1337, EPSILON);

    {
        auto params_temp = params;
        params_temp.dynamics.mass += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.dynamics.J[2][2] += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.dynamics.rotor_thrust_coefficients[3][1] += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.integration.dt += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.init.max_position += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.observation_noise.position += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.action_noise.normalized_rpm += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.reward.position += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.reward.d_action[2] += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.reward.position_clip += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.mdp.termination.position_threshold += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.disturbances.random_force.mean += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.domain_randomization.mass_max += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
    {
        auto params_temp = params;
        params_temp.domain_randomization.rotor_time_constant_falling_max += 1337;
        T diff = rlt::abs_diff(device, params, params_temp);
        ASSERT_NEAR(diff, 1337, EPSILON);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, FROM_JSON_REJECTS_ROTOR_COUNT_MISMATCH){
    using rlt::init;
    using rlt::malloc;
    using rlt::free;
    using rlt::json;
    using rlt::from_json;
    DEVICE device;
    ENVIRONMENT environment;
    init(device);
    malloc(device, environment);
    init(device, environment);
    const auto original = environment.parameters.dynamics;
    const auto serialized = json(device, environment, original);
    const auto valid = nlohmann::json::parse(serialized);
    for(const char* key : {"rotor_positions", "rotor_thrust_directions", "rotor_torque_directions", "rotor_thrust_coefficients", "rotor_torque_constants", "rotor_time_constants_rising", "rotor_time_constants_falling"}){
        for(TI count : {TI(3), TI(6)}){
            auto invalid = valid;
            invalid[key] = nlohmann::json::array();
            for(TI index = 0; index < count; index++){
                invalid[key].push_back(valid[key][0]);
            }
            auto restored = original;
            try {
                from_json(device, environment, invalid, restored);
                FAIL() << key << " accepted " << count << " rotor entries";
            }
            catch(const std::runtime_error& error){
                EXPECT_NE(std::string(error.what()).find(std::string(key) + " has " + std::to_string(count) + " entries, expected 4"), std::string::npos);
            }
            EXPECT_EQ(json(device, environment, restored), serialized);
        }
    }
    free(device, environment);
}

namespace {
    namespace l2f = rlt::rl::environments::l2f;

    template <typename SCALAR, l2f::parameters::dynamics::REGISTRY MODEL>
    void check_estimated_brushless_dynamics(SCALAR mass, SCALAR wheelbase){
        using rlt::abs_diff;
        using rlt::from_json;
        using rlt::init;
        using rlt::json;
        using l2f::multirotor_dynamics;
        using SPEC = l2f::Specification<SCALAR, TI>;
        using STATE = l2f::StateBase<l2f::StateSpecification<SCALAR, TI>>;
        DEVICE device;
        init(device);
        rlt::rl::environments::Multirotor<SPEC> environment;
        auto parameters = environment.parameters;
        parameters.dynamics = l2f::parameters::dynamics::registry<MODEL, SPEC>;
        const auto& dynamics = parameters.dynamics;
        constexpr SCALAR tolerance = 32 * std::numeric_limits<SCALAR>::epsilon();
        EXPECT_NEAR(dynamics.mass, mass, tolerance * mass);
        EXPECT_EQ(dynamics.action_limit.min, SCALAR(0));
        EXPECT_EQ(dynamics.action_limit.max, SCALAR(1));
        SCALAR max_diagonal = 0;
        SCALAR centroid[3] = {};
        for(TI rotor = 0; rotor < 4; rotor++){
            const auto& position = dynamics.rotor_positions[rotor];
            EXPECT_EQ(position[2], SCALAR(0));
            for(TI axis = 0; axis < 3; axis++){
                centroid[axis] += position[axis] / 4;
            }
            for(TI other = 0; other < rotor; other++){
                const SCALAR dx = position[0] - dynamics.rotor_positions[other][0];
                const SCALAR dy = position[1] - dynamics.rotor_positions[other][1];
                max_diagonal = std::max(max_diagonal, std::sqrt(dx * dx + dy * dy));
            }
            EXPECT_EQ(dynamics.rotor_thrust_coefficients[rotor][0], SCALAR(0));
            EXPECT_EQ(dynamics.rotor_thrust_coefficients[rotor][1], SCALAR(0));
            EXPECT_NEAR(dynamics.rotor_torque_constants[rotor], SCALAR(0.004665) * wheelbase / SCALAR(0.08), tolerance);
            EXPECT_NEAR(dynamics.rotor_time_constants_rising[rotor], SCALAR(0.033), tolerance);
            EXPECT_EQ(dynamics.rotor_time_constants_falling[rotor], dynamics.rotor_time_constants_rising[rotor]);
        }
        EXPECT_NEAR(max_diagonal, wheelbase, tolerance * wheelbase);
        for(TI axis = 0; axis < 3; axis++){
            EXPECT_NEAR(centroid[axis], SCALAR(0), tolerance * wheelbase);
        }
        for(TI row = 0; row < 3; row++){
            ASSERT_GT(dynamics.J[row][row], SCALAR(0));
            for(TI column = 0; column < 3; column++){
                EXPECT_EQ(dynamics.J[row][column], dynamics.J[column][row]);
                if(row != column){
                    EXPECT_EQ(dynamics.J[row][column], SCALAR(0));
                }
                SCALAR product = 0;
                for(TI inner = 0; inner < 3; inner++){
                    product += dynamics.J[row][inner] * dynamics.J_inv[inner][column];
                }
                EXPECT_NEAR(product, row == column ? SCALAR(1) : SCALAR(0), tolerance);
            }
        }
        EXPECT_NEAR(2 * dynamics.J[2][2] / (dynamics.J[0][0] + dynamics.J[1][1]), SCALAR(1.832), tolerance);
        l2f::parameters::Dynamics<SCALAR, TI, 4> restored{};
        from_json(device, environment, nlohmann::json::parse(json(device, environment, dynamics)), restored);
        EXPECT_EQ(abs_diff(device, dynamics, restored), SCALAR(0));

        STATE state{};
        state.orientation[0] = 1;
        const SCALAR gravity = SCALAR(9.81);
        const SCALAR hover = SCALAR(1) / std::sqrt(SCALAR(3.5));
        EXPECT_NEAR(dynamics.hovering_throttle_relative, hover, tolerance);
        for(SCALAR throttle : {SCALAR(0), hover, SCALAR(1)}){
            SCOPED_TRACE(throttle);
            SCALAR commands[4] = {throttle, throttle, throttle, throttle};
            STATE derivative{};
            multirotor_dynamics(device, parameters, state, commands, derivative);
            const SCALAR expected_acceleration = throttle == SCALAR(0) ? -gravity : throttle == SCALAR(1) ? SCALAR(2.5) * gravity : SCALAR(0);
            for(TI axis = 0; axis < 3; axis++){
                EXPECT_NEAR(derivative.linear_velocity[axis], axis == 2 ? expected_acceleration : SCALAR(0), 4 * tolerance * gravity);
                EXPECT_NEAR(derivative.angular_velocity[axis], SCALAR(0), 4 * tolerance * gravity);
            }
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, ESTIMATED_BRUSHLESS_DYNAMICS_FLOAT){
    check_estimated_brushless_dynamics<float, l2f::parameters::dynamics::REGISTRY::crazyflie_brushless>(0.0423f, 0.1f);
    check_estimated_brushless_dynamics<float, l2f::parameters::dynamics::REGISTRY::savagebee_pusher>(0.1554f, 0.12f);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, ESTIMATED_BRUSHLESS_DYNAMICS_DOUBLE){
    check_estimated_brushless_dynamics<double, l2f::parameters::dynamics::REGISTRY::crazyflie_brushless>(0.0423, 0.1);
    check_estimated_brushless_dynamics<double, l2f::parameters::dynamics::REGISTRY::savagebee_pusher>(0.1554, 0.12);
}
