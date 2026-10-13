#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/raptor/operations_cpu.h>
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
using PILOT = hd::Raptor<hd::raptor::Specification<rlt::numeric_types::Policy<T>, TI, 1>>;

struct RaptorObservationState {
    T position[3] = {};
    T orientation[4] = {1, 0, 0, 0};
    T linear_velocity[3] = {};
    T angular_velocity[3] = {};
};

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, TargetErrorsAndClipping) {
    DEVICE device;
    PILOT policy;
    malloc(device, policy);
    RaptorObservationState state{{2, -2, T(0.2)}, {1, 0, 0, 0}, {3, -3, 1}, {T(0.1), T(-0.2), T(0.3)}};
    hd::raptor::Target<T> target{{T(1.5), T(-1.5), T(-0.1)}, {T(0.5), T(-0.5), T(0.25)}, 0};
    const T previous[4] = {T(-0.5), T(-0.25), T(0.25), T(0.5)};
    observe(device, policy, state, target, previous, TI(0));
    const T expected[22] = {T(0.5), T(-0.5), T(0.3), 1, 0, 0, 0, 1, 0, 0, 0, 1, 2, -2, T(0.75), T(0.1), T(-0.2), T(0.3), T(-0.5), T(-0.25), T(0.25), T(0.5)};
    for (TI item = 0; item < 22; item++) {
        EXPECT_NEAR(get(device, policy.observations, 0, item), expected[item], 1e-6);
    }
    state.position[0] = 100;
    state.position[1] = -100;
    observe(device, policy, state, target, previous, TI(0));
    EXPECT_EQ(get(device, policy.observations, 0, 0), 1);
    EXPECT_EQ(get(device, policy.observations, 0, 1), -1);
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, QuarterTurnPreservesBodyQuantities) {
    DEVICE device;
    PILOT policy;
    malloc(device, policy);
    const T yaw = rlt::math::PI<T> / 2;
    RaptorObservationState state{{1, T(2.5), T(3.25)}, {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)}, {0, 1, T(0.5)}, {T(0.2), T(-0.4), T(0.6)}};
    hd::raptor::Target<T> target{{1, 2, 3}, {}, yaw};
    const T previous[4] = {T(-0.8), T(-0.2), T(0.4), T(0.9)};
    observe(device, policy, state, target, previous, TI(0));
    EXPECT_NEAR(get(device, policy.observations, 0, 0), T(0.5), 1e-6);
    EXPECT_NEAR(get(device, policy.observations, 0, 1), 0, 1e-6);
    EXPECT_NEAR(get(device, policy.observations, 0, 2), T(0.25), 1e-6);
    EXPECT_NEAR(get(device, policy.observations, 0, 12), 1, 1e-6);
    EXPECT_NEAR(get(device, policy.observations, 0, 13), 0, 1e-6);
    EXPECT_NEAR(get(device, policy.observations, 0, 14), T(0.5), 1e-6);
    for (TI row = 0; row < 3; row++) {
        for (TI col = 0; col < 3; col++) {
            EXPECT_NEAR(get(device, policy.observations, 0, 3 + 3 * row + col), row == col ? 1 : 0, 1e-6);
        }
        EXPECT_EQ(get(device, policy.observations, 0, 15 + row), state.angular_velocity[row]);
    }
    for (TI motor = 0; motor < 4; motor++) {
        EXPECT_EQ(get(device, policy.observations, 0, 18 + motor), previous[motor]);
    }
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, YawWrapIsContinuous) {
    DEVICE device;
    PILOT policy;
    malloc(device, policy);
    const T yaw = -rlt::math::PI<T> + T(0.02);
    RaptorObservationState state{{T(0.2), T(-0.5), T(0.3)}, {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)}, {T(0.2), T(0.7), T(-0.3)}, {1, 2, 3}};
    hd::raptor::Target<T> target{{}, {}, yaw};
    const T previous[4] = {};
    observe(device, policy, state, target, previous, TI(0));
    T before[22];
    for (TI item = 0; item < 22; item++) {
        before[item] = get(device, policy.observations, 0, item);
    }
    target.yaw += 2 * rlt::math::PI<T>;
    observe(device, policy, state, target, previous, TI(0));
    for (TI item = 0; item < 22; item++) {
        EXPECT_NEAR(get(device, policy.observations, 0, item), before[item], 1e-6);
    }
    free(device, policy);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, InferenceMatchesDirectRecurrentCalls) {
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    PILOT policy;
    PILOT::MODEL::State<true> reference_state;
    PILOT::MODEL::Buffer<true> reference_buffer;
    decltype(policy.actions) reference_actions;
    malloc(device, rng);
    init(device, rng, 7);
    malloc(device, policy);
    malloc(device, reference_state);
    malloc(device, reference_buffer);
    malloc(device, reference_actions);
    init_weights(device, policy.model, rng);
    reset(device, policy, rng);
    reset(device, policy.model, reference_state, rng);
    T initial_action[4];
    T max_error = 0;
    for (TI step_i = 0; step_i < 5; step_i++) {
        for (TI item = 0; item < 22; item++) {
            set(device, policy.observations, T(item + step_i) / 22, 0, item);
        }
        evaluate_step(device, policy, rng);
        rlt::Mode<rlt::nn::layers::gru::NoAutoResetMode<rlt::mode::Default<>>> mode;
        evaluate_step(device, policy.model, policy.observations, reference_state, reference_actions, reference_buffer, rng, mode);
        for (TI motor = 0; motor < 4; motor++) {
            T actual = get(device, policy.actions, 0, motor);
            T expected = get(device, reference_actions, 0, motor);
            max_error = rlt::math::max(device.math, max_error, std::abs(actual - expected));
            EXPECT_EQ(actual, expected);
            if (step_i == 0) {
                initial_action[motor] = actual;
            }
        }
    }
    reset(device, policy, rng);
    for (TI item = 0; item < 22; item++) {
        set(device, policy.observations, T(item) / 22, 0, item);
    }
    evaluate_step(device, policy, rng);
    for (TI motor = 0; motor < 4; motor++) {
        EXPECT_EQ(get(device, policy.actions, 0, motor), initial_action[motor]);
    }
    metra::log("hyperdrone/raptor/recurrent_extraction_max_error", max_error);
    free(device, reference_actions);
    free(device, reference_buffer);
    free(device, reference_state);
    free(device, policy);
    free(device, rng);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, YawInputLimitsAndRelease) {
    DEVICE device;
    PILOT policy;
    for (T dt : {T(0.005), T(0.01), T(0.04)}) {
        for (T yaw : {T(0), rlt::math::PI<T> - T(0.005), -rlt::math::PI<T> + T(0.005)}) {
            for (T direction : {T(-1), T(1)}) {
                RaptorObservationState state;
                state.orientation[0] = std::cos(yaw / 2);
                state.orientation[3] = std::sin(yaw / 2);
                hd::raptor::Target<T> target;
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
    constexpr T dt = T(0.01);
    for (T direction : {T(-1), T(1)}) {
        const T yaw = direction * (rlt::math::PI<T> - T(0.005));
        RaptorObservationState state;
        state.orientation[0] = std::cos(yaw / 2);
        state.orientation[3] = std::sin(yaw / 2);
        hd::raptor::Target<T> target;
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
        reset(device, policy, rng);
        FLIGHT_ENV env;
        auto parameters = env.parameters;
        parameters.dynamics = l2f::parameters::dynamics::registry<MODEL, FLIGHT_CONFIG::PARAMETERS_SPEC>;
        FLIGHT_ENV::State state, next;
        initial_state(device, env, parameters, state);
        constexpr auto canonical = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::x500, FLIGHT_CONFIG::PARAMETERS_SPEC>;
        TI motor_order[4] = {};
        for (TI motor = 0; motor < 4; motor++) {
            for (TI policy_motor = 0; policy_motor < 4; policy_motor++) {
                if ((parameters.dynamics.rotor_positions[motor][0] > 0) == (canonical.rotor_positions[policy_motor][0] > 0) && (parameters.dynamics.rotor_positions[motor][1] > 0) == (canonical.rotor_positions[policy_motor][1] > 0)) {
                    motor_order[motor] = policy_motor;
                }
            }
        }
        T previous[4] = {};
        hd::raptor::Target<T> target;
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
            observe(device, policy, state, target, previous, TI(0));
            evaluate_step(device, policy, rng);
            for (TI motor = 0; motor < 4; motor++) {
                previous[motor] = get(device, policy.actions, 0, motor);
            }
            for (TI motor = 0; motor < 4; motor++) {
                set(action, 0, motor, previous[motor_order[motor]]);
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
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR, CheckpointFlightAcrossDroneModels) {
    if (std::getenv("CONTA_ROOT") == nullptr) {
        GTEST_SKIP() << "Set CONTA_ROOT to a local store containing the pinned RAPTOR checkpoint";
    }
    std::string path, error;
    if (!conta::resolve(hd::raptor::CHECKPOINT + 6, path, error)) {
        GTEST_SKIP() << error;
    }
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    PILOT policy;
    malloc(device, rng);
    init(device, rng, 0);
    malloc(device, policy);
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
    free(device, policy);
    free(device, rng);
}
