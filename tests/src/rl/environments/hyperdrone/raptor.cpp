#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/raptor/operations_cpu.h>
#include <conta/conta.h>
#include <metra/metra.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>

namespace rlt = rl_tools;
namespace hd = rlt::rl::environments::hyperdrone;
using namespace rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = DEVICE::index_t;
using PILOT = hd::controllers::raptor::Config::Bind<rlt::numeric_types::Policy<T>, TI, 1>;

struct RaptorObservationState {
    T position[3] = {};
    T orientation[4] = {1, 0, 0, 0};
    T linear_velocity[3] = {};
    T angular_velocity[3] = {};
};

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, TargetErrorsAndClipping) {
    DEVICE device;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    malloc(device, policy);
    malloc(device, controller_state);
    malloc(device, buffer);
    RaptorObservationState state{{2, -2, T(0.2)}, {1, 0, 0, 0}, {3, -3, 1}, {T(0.1), T(-0.2), T(0.3)}};
    hd::controllers::raptor::Target<T> target{{T(1.5), T(-1.5), T(-0.1)}, {T(0.5), T(-0.5), T(0.25)}, 0};
    const T previous[4] = {T(-0.5), T(-0.25), T(0.25), T(0.5)};
    observe(device, policy, buffer, state, target, previous, TI(0));
    const T expected[22] = {T(0.5), T(-0.5), T(0.3), 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, -2, T(0.75), T(0.1), T(-0.2), T(0.3), T(-0.5), T(-0.25), T(0.25), T(0.5)};
    for (TI item = 0; item < 22; item++) {
        EXPECT_NEAR(get(device, buffer.observations, 0, item), expected[item], 1e-6);
    }
    state.position[0] = 100;
    state.position[1] = -100;
    observe(device, policy, buffer, state, target, previous, TI(0));
    EXPECT_EQ(get(device, buffer.observations, 0, 0), 1);
    EXPECT_EQ(get(device, buffer.observations, 0, 1), -1);
    free(device, buffer);
    free(device, controller_state);
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, QuarterTurnPreservesBodyQuantities) {
    DEVICE device;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    malloc(device, policy);
    malloc(device, controller_state);
    malloc(device, buffer);
    const T yaw = rlt::math::PI<T> / 2;
    RaptorObservationState state{{1, T(2.5), T(3.25)}, {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)}, {0, 1, T(0.5)}, {T(0.2), T(-0.4), T(0.6)}};
    hd::controllers::raptor::Target<T> target{{1, 2, 3}, {}, yaw};
    const T previous[4] = {T(-0.8), T(-0.2), T(0.4), T(0.9)};
    observe(device, policy, buffer, state, target, previous, TI(0));
    EXPECT_NEAR(get(device, buffer.observations, 0, 0), T(0.5), 1e-6);
    EXPECT_NEAR(get(device, buffer.observations, 0, 1), 0, 1e-6);
    EXPECT_NEAR(get(device, buffer.observations, 0, 2), T(0.25), 1e-6);
    EXPECT_NEAR(get(device, buffer.observations, 0, 12), 1, 1e-6);
    EXPECT_NEAR(get(device, buffer.observations, 0, 13), 0, 1e-6);
    EXPECT_NEAR(get(device, buffer.observations, 0, 14), T(0.5), 1e-6);
    for (TI row = 0; row < 3; row++) {
        for (TI col = 0; col < 3; col++) {
            EXPECT_NEAR(get(device, buffer.observations, 0, 3 + 3 * row + col), row == col ? 1 : 0, 1e-6);
        }
        EXPECT_EQ(get(device, buffer.observations, 0, 15 + row), state.angular_velocity[row]);
    }
    for (TI motor = 0; motor < 4; motor++) {
        EXPECT_EQ(get(device, buffer.observations, 0, 18 + motor), previous[motor]);
    }
    free(device, buffer);
    free(device, controller_state);
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, YawWrapIsContinuous) {
    DEVICE device;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    malloc(device, policy);
    malloc(device, controller_state);
    malloc(device, buffer);
    const T yaw = -rlt::math::PI<T> + T(0.02);
    RaptorObservationState state{{T(0.2), T(-0.5), T(0.3)}, {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)}, {T(0.2), T(0.7), T(-0.3)}, {1, 2, 3}};
    hd::controllers::raptor::Target<T> target{{}, {}, yaw};
    const T previous[4] = {};
    observe(device, policy, buffer, state, target, previous, TI(0));
    T before[22];
    for (TI item = 0; item < 22; item++) {
        before[item] = get(device, buffer.observations, 0, item);
    }
    target.yaw += 2 * rlt::math::PI<T>;
    observe(device, policy, buffer, state, target, previous, TI(0));
    for (TI item = 0; item < 22; item++) {
        EXPECT_NEAR(get(device, buffer.observations, 0, item), before[item], 1e-6);
    }
    free(device, buffer);
    free(device, controller_state);
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, InferenceMatchesDirectRecurrentCalls) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    PILOT::MODEL::State<true> reference_state;
    PILOT::MODEL::Buffer<true> reference_buffer;
    decltype(controller_state.previous_actions) reference_actions;
    malloc(device, rng);
    init(device, rng, 7);
    malloc(device, policy);
    malloc(device, controller_state);
    malloc(device, buffer);
    malloc(device, reference_state);
    malloc(device, reference_buffer);
    malloc(device, reference_actions);
    init_weights(device, policy.model, rng);
    reset(device, policy, controller_state, rng);
    reset(device, policy.model, reference_state, rng);
    T initial_action[4];
    T max_error = 0;
    for (TI step_i = 0; step_i < 5; step_i++) {
        for (TI item = 0; item < 22; item++) {
            set(device, buffer.observations, T(item + step_i) / 22, 0, item);
        }
        evaluate_step(device, policy, controller_state, buffer, rng);
        rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>> mode;
        evaluate_step(device, policy.model, buffer.observations, reference_state, reference_actions, reference_buffer, rng, mode);
        for (TI motor = 0; motor < 4; motor++) {
            T actual = get(device, controller_state.previous_actions, 0, motor);
            T expected = get(device, reference_actions, 0, motor);
            max_error = rlt::math::max(device.math, max_error, std::abs(actual - expected));
            EXPECT_EQ(actual, expected);
            if (step_i == 0) {
                initial_action[motor] = actual;
            }
        }
    }
    reset(device, policy, controller_state, rng);
    for (TI item = 0; item < 22; item++) {
        set(device, buffer.observations, T(item) / 22, 0, item);
    }
    evaluate_step(device, policy, controller_state, buffer, rng);
    for (TI motor = 0; motor < 4; motor++) {
        EXPECT_EQ(get(device, controller_state.previous_actions, 0, motor), initial_action[motor]);
    }
    metra::log("hyperdrone/raptor/recurrent_extraction_max_error", max_error);
    free(device, reference_actions);
    free(device, reference_buffer);
    free(device, reference_state);
    free(device, buffer);
    free(device, controller_state);
    free(device, policy);
    free(device, rng);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, YawInputLimitsAndRelease) {
    DEVICE device;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    for (T dt : {T(0.005), T(0.01), T(0.04)}) {
        for (T yaw : {T(0), rlt::math::PI<T> - T(0.005), -rlt::math::PI<T> + T(0.005)}) {
            for (T direction : {T(-1), T(1)}) {
                RaptorObservationState state;
                state.orientation[0] = std::cos(yaw / 2);
                state.orientation[3] = std::sin(yaw / 2);
                hd::controllers::raptor::Target<T> target;
                target.yaw = yaw;
                for (TI step_i = 0; step_i < 100; step_i++) {
                    const T before = target.yaw;
                    update_yaw_target(device, policy, state, target, direction * T(100), dt);
                    const T delta = std::atan2(std::sin(target.yaw - before), std::cos(target.yaw - before));
                    const T error = std::atan2(std::sin(target.yaw - yaw), std::cos(target.yaw - yaw));
                    EXPECT_GE(direction * delta, -1e-6);
                    EXPECT_LE(std::abs(delta), PILOT::SPEC::YAW_RATE_LIMIT * dt + 1e-6);
                    EXPECT_LE(std::abs(error), PILOT::SPEC::YAW_ERROR_LIMIT + 1e-6);
                }
                const T held = target.yaw;
                state.orientation[0] = std::cos(held / 2);
                state.orientation[3] = std::sin(held / 2);
                for (TI step_i = 0; step_i < 100; step_i++) {
                    update_yaw_target(device, policy, state, target, T(0), dt);
                    EXPECT_NEAR(target.yaw, held, 1e-6);
                }
                update_yaw_target(device, policy, state, target, -direction * T(100), dt);
                const T reverse_delta = std::atan2(std::sin(target.yaw - held), std::cos(target.yaw - held));
                EXPECT_NEAR(reverse_delta, -direction * PILOT::SPEC::YAW_RATE_LIMIT * dt, 1e-6);
            }
        }
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, RuntimeYawLimits) {
    DEVICE device;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    constexpr T dt = T(0.01);
    for (T direction : {T(-1), T(1)}) {
        const T yaw = direction * (rlt::math::PI<T> - T(0.005));
        RaptorObservationState state;
        state.orientation[0] = std::cos(yaw / 2);
        state.orientation[3] = std::sin(yaw / 2);
        hd::controllers::raptor::Target<T> target;
        target.yaw = yaw;
        update_yaw_target(device, policy, state, target, direction * T(100), dt, T(3), T(0.6));
        EXPECT_NEAR(std::atan2(std::sin(target.yaw - yaw), std::cos(target.yaw - yaw)), direction * T(0.03), 1e-6);
        for (TI step_i = 0; step_i < 100; step_i++) {
            update_yaw_target(device, policy, state, target, direction * T(100), dt, T(3), T(0.6));
        }
        EXPECT_NEAR(std::atan2(std::sin(target.yaw - yaw), std::cos(target.yaw - yaw)), direction * T(0.6), 1e-6);
        update_yaw_target(device, policy, state, target, T(0), dt, T(0.2), T(0.1));
        EXPECT_NEAR(std::atan2(std::sin(target.yaw - yaw), std::cos(target.yaw - yaw)), direction * T(0.1), 1e-6);
        const T before = target.yaw;
        update_yaw_target(device, policy, state, target, -direction * T(100), dt, T(0.2), T(0.1));
        EXPECT_NEAR(std::atan2(std::sin(target.yaw - before), std::cos(target.yaw - before)), -direction * T(0.002), 1e-6);
    }
}

namespace {
    namespace l2f = rlt::rl::environments::l2f;
    using FLIGHT_CONFIG = hd::presets::x500_fpv::DynamicsStaticParameters<T, TI>;
    using FLIGHT_ENV = rlt::rl::environments::Multirotor<l2f::Specification<T, TI, FLIGHT_CONFIG>>;

    template <l2f::parameters::dynamics::REGISTRY MODEL, typename YAW_INPUT>
    void fly(DEVICE& device, PILOT& policy, DEVICE::SPEC::RANDOM::ENGINE<>& rng, T turn_direction, YAW_INPUT yaw_input) {
        PILOT::State controller_state;
        PILOT::Buffer buffer;
        malloc(device, controller_state);
        malloc(device, buffer);
        reset(device, policy, controller_state, rng);
        FLIGHT_ENV env;
        auto parameters = env.parameters;
        parameters.dynamics = l2f::parameters::dynamics::registry<MODEL, FLIGHT_CONFIG::PARAMETERS_SPEC>;
        FLIGHT_ENV::State state, next;
        initial_state(device, env, parameters, state);
        hd::controllers::raptor::Target<T> target;
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, 4, false>> action;
        T max_altitude_error = 0;
        T max_body_yaw_rate = 0;
        T forward_displacement = 0;
        for (TI step_i = 0; step_i < 1100; step_i++) {
            const bool translating = (step_i >= 100 && step_i < 300) || (step_i >= 500 && step_i < 700);
            update_yaw_target(device, policy, state, target, turn_direction * yaw_input(step_i), parameters.integration.dt);
            if (translating || step_i == 300 || step_i == 700) {
                const auto& q = state.orientation;
                const T heading = std::atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] * q[2] + q[3] * q[3]));
                for (TI axis = 0; axis < 2; axis++) {
                    const T direction = axis == 0 ? std::cos(heading) : std::sin(heading);
                    target.position[axis] = state.position[axis] + (translating ? T(0.5) * direction : 0);
                    target.linear_velocity[axis] = translating ? T(0.6) * direction : 0;
                }
            }
            observe(device, policy, controller_state, buffer, state, target, TI(0));
            evaluate_step(device, policy, controller_state, buffer, rng);
            for (TI motor = 0; motor < 4; motor++) {
                set(action, 0, motor, get(device, controller_state.previous_actions, 0, motor));
            }
            ASSERT_EQ(step(device, env, parameters, state, action, next, rng), T(0.01));
            state = next;
            ASSERT_TRUE(std::isfinite(state.position[2]));
            ASSERT_TRUE(std::isfinite(state.angular_velocity[2]));
            max_altitude_error = rlt::math::max(device.math, max_altitude_error, std::abs(state.position[2]));
            max_body_yaw_rate = rlt::math::max(device.math, max_body_yaw_rate, std::abs(state.angular_velocity[2]));
            if (step_i == 299) {
                forward_displacement = state.position[0];
            }
        }
        const T yaw = std::atan2(2 * (state.orientation[0] * state.orientation[3] + state.orientation[1] * state.orientation[2]), 1 - 2 * (state.orientation[2] * state.orientation[2] + state.orientation[3] * state.orientation[3]));
        const T speed = std::sqrt(state.linear_velocity[0] * state.linear_velocity[0] + state.linear_velocity[1] * state.linear_velocity[1] + state.linear_velocity[2] * state.linear_velocity[2]);
        const T yaw_error = std::atan2(std::sin(yaw - target.yaw), std::cos(yaw - target.yaw));
        EXPECT_TRUE(std::isfinite(speed));
        EXPECT_LT(speed, T(0.1));
        EXPECT_LT(std::abs(yaw_error), T(0.1));
        EXPECT_LT(max_altitude_error, T(0.5));
        EXPECT_LT(max_body_yaw_rate, T(2.5));
        EXPECT_GT(forward_displacement, T(0.2));
        EXPECT_GT(turn_direction * state.position[1], T(0.2));
        metra::log("hyperdrone/raptor/flight/final_speed", speed);
        metra::log("hyperdrone/raptor/flight/yaw_error", std::abs(yaw_error));
        metra::log("hyperdrone/raptor/flight/max_altitude_error", max_altitude_error);
        metra::log("hyperdrone/raptor/flight/max_body_yaw_rate", max_body_yaw_rate);
        free(device, buffer);
        free(device, controller_state);
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, CheckpointFlightAcrossDroneModels) {
    if (std::getenv("CONTA_ROOT") == nullptr) {
        GTEST_SKIP() << "Set CONTA_ROOT to a local store containing the pinned RAPTOR checkpoint";
    }
    std::string path, error;
    if (!conta::resolve(hd::controllers::raptor::CHECKPOINT + 6, path, error)) {
        GTEST_SKIP() << error;
    }
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    PILOT policy;
    PILOT::State controller_state;
    PILOT::Buffer buffer;
    malloc(device, rng);
    init(device, rng, 0);
    malloc(device, policy);
    malloc(device, controller_state);
    malloc(device, buffer);
    ASSERT_TRUE(load(device, policy, path));
    struct YawScenario {
        const char* name;
        T (*input)(TI);
    };
    const YawScenario scenarios[] = {
        {"smooth", [](TI step_i) -> T {
            return step_i >= 300 && step_i < 500 ? rlt::math::PI<T> / 400 : 0;
        }},
        {"rapid reversals", [](TI step_i) -> T {
            if (step_i >= 300 && step_i < 400) {
                return (step_i / 4) % 2 == 0 ? T(100) : T(-100);
            }
            if (step_i >= 400 && step_i < 650) {
                return T(100);
            }
            return step_i >= 650 && step_i < 700 ? T(-100) : 0;
        }},
    };
    for (const auto& scenario : scenarios) {
        SCOPED_TRACE(scenario.name);
        for (T direction : {T(-1), T(1)}) {
            SCOPED_TRACE(direction);
            {
                SCOPED_TRACE("x500");
                fly<l2f::parameters::dynamics::REGISTRY::x500>(device, policy, rng, direction, scenario.input);
            }
            {
                SCOPED_TRACE("crazyflie");
                fly<l2f::parameters::dynamics::REGISTRY::crazyflie>(device, policy, rng, direction, scenario.input);
            }
            {
                SCOPED_TRACE("crazyflie_brushless");
                fly<l2f::parameters::dynamics::REGISTRY::crazyflie_brushless>(device, policy, rng, direction, scenario.input);
            }
            {
                SCOPED_TRACE("savagebee_pusher");
                fly<l2f::parameters::dynamics::REGISTRY::savagebee_pusher>(device, policy, rng, direction, scenario.input);
            }
            {
                SCOPED_TRACE("arpl");
                fly<l2f::parameters::dynamics::REGISTRY::arpl>(device, policy, rng, direction, scenario.input);
            }
            {
                SCOPED_TRACE("soft");
                fly<l2f::parameters::dynamics::REGISTRY::soft>(device, policy, rng, direction, scenario.input);
            }
        }
    }
    free(device, buffer);
    free(device, controller_state);
    free(device, policy);
    free(device, rng);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, FlatBatchReturnsBorrowedActions) {
    using CONTROLLER = hd::controllers::raptor::Config::Bind<rlt::numeric_types::Policy<T>, TI, 3>;
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    CONTROLLER controller;
    CONTROLLER::State state, reference;
    CONTROLLER::Buffer buffer, reference_buffer;
    rlt::Tensor<rlt::tensor::Specification<FLIGHT_ENV::Parameters, TI, rlt::tensor::Shape<TI, 3>>> parameters;
    rlt::Tensor<rlt::tensor::Specification<FLIGHT_ENV::State, TI, rlt::tensor::Shape<TI, 3>>> physical;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 3, 4>>> commands;
    malloc(device, rng);
    init(device, rng, 11);
    malloc(device, controller);
    malloc(device, state);
    malloc(device, reference);
    malloc(device, buffer);
    malloc(device, reference_buffer);
    malloc(device, parameters);
    malloc(device, physical);
    malloc(device, commands);
    init_weights(device, controller.model, rng);
    reset(device, controller, state, rng);
    reset(device, controller, reference, rng);
    FLIGHT_ENV dynamics;
    for (TI row = 0; row < 3; row++) {
        get_ref(device, parameters, row) = dynamics.parameters;
        initial_state(device, dynamics, get_ref(device, parameters, row), get_ref(device, physical, row));
        hd::controllers::raptor::Target<T> target;
        for (TI axis = 0; axis < 3; axis++) {
            target.position[axis] = T(row + axis) / 10;
            set(device, commands, target.position[axis], row, axis);
        }
        target.yaw = T(row) / 4;
        set(device, commands, target.yaw, row, 3);
        observe(device, controller, reference, reference_buffer, get_ref(device, physical, row), target, row);
    }
    auto actual = control(device, controller, state, buffer, parameters, physical, commands, rng);
    evaluate_step(device, controller, reference, reference_buffer, rng);
    EXPECT_EQ(data(actual), data(state.previous_actions));
    EXPECT_LT(abs_diff(device, actual, reference.previous_actions), 1e-6);
    EXPECT_EQ(abs_diff(device, buffer.observations, reference_buffer.observations), 0);
    free(device, commands);
    free(device, physical);
    free(device, parameters);
    free(device, reference_buffer);
    free(device, buffer);
    free(device, reference);
    free(device, state);
    free(device, controller);
    free(device, rng);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, ExpandedMaskedResetPreservesOtherInstances) {
    using CONTROLLER = hd::controllers::raptor::Config::Bind<rlt::numeric_types::Policy<T>, TI, 4>;
    using rlt::nn_models::sequential::content_state;
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    CONTROLLER controller;
    CONTROLLER::State state, before;
    CONTROLLER::Buffer buffer;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, 2>>> mask;
    malloc(device, rng);
    init(device, rng, 13);
    malloc(device, controller);
    malloc(device, state);
    malloc(device, before);
    malloc(device, buffer);
    malloc(device, mask);
    init_weights(device, controller.model, rng);
    set_all(device, get_layer<1>(controller.model).initial_hidden_state.parameters, T(0.25));
    reset(device, controller, state, rng);
    set_all(device, buffer.observations, T(0.1));
    for (TI step = 0; step < 20; step++) {
        evaluate_step(device, controller, state, buffer, rng);
    }
    copy(device, device, state, before);
    set(device, mask, true, 0);
    set(device, mask, false, 1);
    reset(device, controller, state, buffer, mask, rng);
    const auto& hidden = content_state<1>(state.recurrent.content_state);
    const auto& previous_hidden = content_state<1>(before.recurrent.content_state);
    for (TI row = 0; row < 4; row++) {
        EXPECT_EQ(get(device, hidden.step, row), row < 2 ? TI(0) : TI(20));
        for (TI feature = 0; feature < 16; feature++) {
            EXPECT_EQ(get(device, hidden.state, row, feature), row < 2 ? T(0.25) : get(device, previous_hidden.state, row, feature));
        }
        for (TI motor = 0; motor < 4; motor++) {
            EXPECT_EQ(get(device, state.previous_actions, row, motor), row < 2 ? T(0) : get(device, before.previous_actions, row, motor));
        }
    }
    free(device, mask);
    free(device, buffer);
    free(device, before);
    free(device, state);
    free(device, controller);
    free(device, rng);
}

struct RaptorAdapterParameters {
    T offset;
};
struct RaptorAdapterState {
    RaptorObservationState agents[2];
};
namespace rl_tools {
    template <typename DEVICE_T, typename CONTROLLER, typename CONTROL_STATE, typename BUFFER, typename COMMAND_SPEC>
    void observe(DEVICE_T& device, const CONTROLLER& controller, const CONTROL_STATE& state, BUFFER& buffer, const RaptorAdapterParameters& parameters, const RaptorAdapterState& physical, const Tensor<COMMAND_SPEC>& commands, TI row, TI agent) {
        typename CONTROLLER::Target target;
        for (TI axis = 0; axis < 3; axis++) {
            target.position[axis] = get(device, commands, agent * 4 + axis) - parameters.offset;
        }
        target.yaw = get(device, commands, agent * 4 + 3);
        observe(device, controller, state, buffer, physical.agents[agent], target, row);
    }
}
TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, LateLayoutAdapterBatchesAgentsWithStaticStorage) {
    using CONTROLLER = hd::controllers::raptor::Config::Bind<rlt::numeric_types::Policy<T>, TI, 4, false>;
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    CONTROLLER controller;
    CONTROLLER::State state;
    CONTROLLER::Buffer buffer;
    rlt::Tensor<rlt::tensor::Specification<RaptorAdapterParameters, TI, rlt::tensor::Shape<TI, 2>, false>> parameters;
    rlt::Tensor<rlt::tensor::Specification<RaptorAdapterState, TI, rlt::tensor::Shape<TI, 2>, false>> physical;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 2, 8>, false>> commands;
    malloc(device, rng);
    init(device, rng, 17);
    malloc(device, controller);
    malloc(device, state);
    malloc(device, buffer);
    init_weights(device, controller.model, rng);
    reset(device, controller, state, rng);
    for (TI instance = 0; instance < 2; instance++) {
        get_ref(device, parameters, instance).offset = T(instance + 1);
        get_ref(device, physical, instance) = {};
        for (TI agent = 0; agent < 2; agent++) {
            for (TI axis = 0; axis < 3; axis++) {
                set(device, commands, T(instance + 1) + T(agent + 1) / 10, instance, agent * 4 + axis);
            }
            set(device, commands, T(0), instance, agent * 4 + 3);
        }
    }
    auto actual = control(device, controller, state, buffer, parameters, physical, commands, rng);
    EXPECT_EQ(data(actual), data(state.previous_actions));
    for (TI row = 0; row < 4; row++) {
        for (TI axis = 0; axis < 3; axis++) {
            EXPECT_NEAR(get(device, buffer.observations, row, axis), -T(row % 2 + 1) / 10, 1e-6);
        }
    }
    free(device, buffer);
    free(device, state);
    free(device, controller);
    free(device, rng);
}
