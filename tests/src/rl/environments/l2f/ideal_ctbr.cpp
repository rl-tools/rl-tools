#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/direct/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/ctbr/operations_cpu.h>
#include <metra/metra.h>
#include <gtest/gtest.h>
#include <cmath>
#include <type_traits>

namespace rlt = rl_tools;
namespace l2f = rlt::rl::environments::l2f;
namespace controllers = rlt::rl::environments::hyperdrone::controllers;
using DEVICE = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t;
using T = double;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using FACTORY = l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;
using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
using STATE_ACCEL = l2f::StateLinearAcceleration<l2f::StateSpecification<T, TI, STATE_BASE>>;
using STATE_IMU = l2f::StateIMU<T, TI, STATE_ACCEL>;
using STATE_LAST = l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_IMU>>;
using STATE_FORCE = l2f::StateRandomForce<l2f::StateSpecification<T, TI, STATE_LAST>>;
using STATE_HISTORY = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, 2, false, STATE_FORCE>>;
using STATE_RENDER = l2f::StateRenderRotorPhase<l2f::StateSpecification<T, TI, STATE_HISTORY>>;

template<typename STATE = STATE_RENDER, TI SUBSTEPS = 1>
struct StaticParameters: controllers::ctbr::Config::DynamicsParameters<FACTORY::STATIC_PARAMETERS>{
    using STATE_TYPE = STATE;
    static constexpr TI N_SUBSTEPS = SUBSTEPS;
};
template<typename STATE = STATE_RENDER, TI SUBSTEPS = 1>
using Environment = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, StaticParameters<STATE, SUBSTEPS>>>;
using ACTION = rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>>;
using rlt::init;
using rlt::malloc;
using rlt::free;
using rlt::initial_parameters;
using rlt::initial_state;
using rlt::step;
using rlt::control;
using rlt::command_bounds;
using rlt::set;
using rlt::get;
using rlt::data;

template<typename ENV = Environment<>>
struct Fixture{
    DEVICE device;
    ENV env;
    typename ENV::Parameters parameters;
    typename ENV::State state{}, next{};
    ACTION action{};
    RNG rng;
    Fixture(){
        init(device);
        malloc(device, rng);
        init(device, rng, 7);
        init(device, env);
        initial_parameters(device, env, parameters);
        parameters.dynamics.mass = 1;
        parameters.dynamics.gravity[0] = 0;
        parameters.dynamics.gravity[1] = 0;
        parameters.dynamics.gravity[2] = -9.81;
        parameters.dynamics.action_limit = {0, 1};
        parameters.dynamics.max_rpm = 1000;
        for(TI rotor = 0; rotor < 4; rotor++){
            parameters.dynamics.rotor_thrust_coefficients[rotor][0] = 0;
            parameters.dynamics.rotor_thrust_coefficients[rotor][1] = 0;
            parameters.dynamics.rotor_thrust_coefficients[rotor][2] = 4;
        }
        parameters.integration.dt = 0.01;
        parameters.imu = {};
        parameters.disturbances = {};
        parameters.mdp.action_noise.normalized_rpm = 100;
        initial_state(device, env, parameters, state);
    }
    ~Fixture(){
        free(device, rng);
    }
    void command(T thrust, T x = 0, T y = 0, T z = 0){
        set(action, 0, 0, thrust);
        set(action, 0, 1, x);
        set(action, 0, 2, y);
        set(action, 0, 3, z);
    }
    void advance(){
        step(device, env, parameters, state, action, next, rng);
    }
};

TEST(L2F_IDEAL_CTBR, HOVER_FREE_FALL_AND_IMU){
    Fixture f;
    f.command(9.81);
    f.advance();
    EXPECT_NEAR(f.next.position[2], 0, 1e-14);
    EXPECT_NEAR(f.next.linear_velocity[2], 0, 1e-14);
    EXPECT_NEAR(f.next.accelerometer[2], 9.81, 1e-12);
    f.command(0);
    f.advance();
    EXPECT_NEAR(f.next.position[2], -9.81 * 0.01 * 0.01 / 2, 1e-12);
    EXPECT_NEAR(f.next.linear_velocity[2], -9.81 * 0.01, 1e-12);
    EXPECT_NEAR(f.next.linear_acceleration[2], -9.81, 1e-12);
    EXPECT_NEAR(f.next.accelerometer[2], 0, 1e-12);
}

TEST(L2F_IDEAL_CTBR, RATES_APPLY_TO_CURRENT_INTERVAL){
    Fixture f;
    f.parameters.integration.dt = 0.1;
    f.state.angular_velocity[0] = -10;
    f.command(9.81, 2);
    f.advance();
    EXPECT_EQ(f.next.angular_velocity[0], 2);
    EXPECT_EQ(f.next.gyro[0], 2);
    EXPECT_NEAR(f.next.orientation[0], std::cos(0.1), 1e-7);
    EXPECT_NEAR(f.next.orientation[1], std::sin(0.1), 1e-7);
    EXPECT_NEAR(f.next.linear_velocity[1], 9.81 * (std::cos(0.2) - 1) / 2, 3e-7);
    EXPECT_NEAR(f.next.linear_velocity[2], 9.81 * (std::sin(0.2) / 2 - 0.1), 4e-6);
    Fixture<Environment<STATE_RENDER, 10>> refined;
    refined.parameters.integration.dt = 0.1;
    refined.command(9.81, 2);
    refined.advance();
    EXPECT_NEAR(refined.next.linear_velocity[2], 9.81 * (std::sin(0.2) / 2 - 0.1), 1e-9);
    f.state = f.next;
    f.command(9.81, -2);
    f.advance();
    EXPECT_NEAR(f.next.orientation[0], 1, 1e-12);
    EXPECT_NEAR(f.next.orientation[1], 0, 1e-12);
    metra::log("l2f/ideal_ctbr/rate_rotation_error", std::abs(f.next.orientation[1]));
}

TEST(L2F_IDEAL_CTBR, FORCE_APPLIES_TORQUE_AND_ROTORS_DO_NOT){
    Fixture a, b;
    a.command(9.81, 1, 2, 3);
    b.command(9.81, 1, 2, 3);
    b.state.force[0] = 2;
    b.state.torque[0] = 100;
    b.state.torque[1] = -500;
    for(TI rotor = 0; rotor < 4; rotor++){
        b.state.rpm[rotor] = 1000;
        b.parameters.dynamics.rotor_time_constants_rising[rotor] = 0;
        b.parameters.dynamics.rotor_time_constants_falling[rotor] = 0;
    }
    b.parameters.dynamics.J_inv[0][0] = 1e10;
    a.advance();
    b.advance();
    EXPECT_NEAR(b.next.linear_velocity[0] - a.next.linear_velocity[0], 0.02, 1e-12);
    for(TI axis = 0; axis < 3; axis++){
        EXPECT_EQ(a.next.angular_velocity[axis], b.next.angular_velocity[axis]);
    }
    for(TI element = 0; element < 4; element++){
        EXPECT_EQ(a.next.orientation[element], b.next.orientation[element]);
    }
}

TEST(L2F_IDEAL_CTBR, PHYSICAL_LIMITS_HISTORY_AND_REWARD){
    Fixture f;
    f.command(100, 100, -100, 4);
    f.advance();
    EXPECT_EQ(f.next.last_action[0], 16);
    EXPECT_EQ(f.next.last_action[1], 20);
    EXPECT_EQ(f.next.last_action[2], -20);
    EXPECT_EQ(f.next.last_action[3], 4);
    for(TI component = 0; component < 4; component++){
        EXPECT_EQ(f.next.action_history[0][component], f.next.last_action[component]);
        EXPECT_EQ(f.state.action_history[0][component], 0);
        EXPECT_EQ(f.next.rpm[component], 1);
    }
    typename FACTORY::REWARD_FUNCTION::Components components{};
    l2f::parameters::reward_functions::reward_components(f.device, f.env, f.parameters, f.parameters.mdp.reward, f.state, f.action, f.next, components, f.rng);
    EXPECT_NEAR(components.action_costs[0], (16 - 9.81) * (16 - 9.81), 1e-10);
    EXPECT_EQ(components.action_costs[1], 400);
    EXPECT_EQ(components.d_action_costs[0], 256);
    for(TI rotor = 0; rotor < 4; rotor++){
        f.parameters.dynamics.rotor_thrust_coefficients[rotor][2] = 5;
    }
    f.advance();
    EXPECT_EQ(f.next.last_action[0], 20);
    f.command(-5);
    f.advance();
    EXPECT_EQ(f.next.last_action[0], 0);
}

TEST(L2F_IDEAL_CTBR, OPTIONAL_ROTORS_AND_SUBSTEPS){
    Fixture<Environment<STATE_FORCE, 10>> lean;
    Fixture<Environment<STATE_RENDER, 10>> full;
    lean.command(9.81, 2, -1, 3);
    full.command(9.81, 2, -1, 3);
    lean.advance();
    full.advance();
    for(TI axis = 0; axis < 3; axis++){
        EXPECT_EQ(lean.next.position[axis], full.next.position[axis]);
        EXPECT_EQ(lean.next.linear_velocity[axis], full.next.linear_velocity[axis]);
        EXPECT_EQ(lean.next.angular_velocity[axis], full.next.angular_velocity[axis]);
    }
    for(TI element = 0; element < 4; element++){
        EXPECT_EQ(lean.next.orientation[element], full.next.orientation[element]);
    }
    EXPECT_TRUE(lean.rng.engine == full.rng.engine);
}

TEST(HYPERDRONE_CONTROLLERS, STATELESS_BORROWS_STATIC_AND_STRIDED_COMMANDS){
    using CTBR = controllers::ctbr::Config::Bind<rlt::numeric_types::Policy<T>, TI, 4, false>;
    using DIRECT = controllers::direct::Config<>::Bind<rlt::numeric_types::Policy<T>, TI, 4, false>;
    static_assert(std::is_empty_v<CTBR::State> && std::is_empty_v<CTBR::Buffer>);
    static_assert(std::is_empty_v<DIRECT::State> && std::is_empty_v<DIRECT::Buffer>);
    using SHAPE = rlt::tensor::Shape<TI, 2, 8>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, SHAPE, false, rlt::tensor::Stride<TI, 16, 2>>> commands{};
    rlt::Tensor<rlt::tensor::Specification<int, TI, rlt::tensor::Shape<TI, 2>, false>> parameters{}, states{};
    using PD = controllers::direct::Config<l2f::parameters::ActionInterface::CTBR>;
    static_assert(PD::DynamicsParameters<FACTORY::STATIC_PARAMETERS>::ACTION_INTERFACE == l2f::parameters::ActionInterface::CTBR);
    DIRECT direct;
    DIRECT::State direct_state;
    DIRECT::Buffer direct_buffer;
    CTBR controller;
    CTBR::State state;
    CTBR::Buffer buffer;
    DEVICE device;
    RNG rng;
    auto actions = control(device, controller, state, buffer, parameters, states, commands, rng);
    auto direct_actions = control(device, direct, direct_state, direct_buffer, parameters, states, commands, rng);
    EXPECT_EQ(data(direct_actions), data(commands));
    EXPECT_EQ(data(actions), data(commands));
    set(device, commands, 13.0, 1, 7);
    EXPECT_EQ(get(device, actions, 1, 7), 13.0);
    constexpr auto collective = command_bounds(CTBR{}, 0);
    constexpr auto rate = command_bounds(CTBR{}, 1);
    EXPECT_EQ(collective.lower, 0);
    EXPECT_EQ(collective.upper, std::numeric_limits<T>::infinity());
    EXPECT_EQ(rate.lower, -20);
    EXPECT_EQ(rate.upper, 20);
}
