#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/registry.h>
#include <gtest/gtest.h>
#include <cmath>

namespace {
    namespace rlt = rl_tools;
    namespace l2f = rlt::rl::environments::l2f;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using T = double;
    using SPEC = l2f::Specification<T, TI>;
    using REGISTRY = l2f::parameters::dynamics::REGISTRY;

    template<REGISTRY MODEL>
    void check_allocation(){
        using rlt::is_canonical_quadrotor;
        using l2f::multirotor_dynamics;
        DEVICE device;
        rlt::rl::environments::Multirotor<SPEC> environment;
        auto parameters = environment.parameters;
        parameters.dynamics = l2f::parameters::dynamics::registry<MODEL, SPEC>;
        const auto& dynamics = parameters.dynamics;
        EXPECT_TRUE(is_canonical_quadrotor(device, dynamics));
        constexpr T signs[4][3] = {{1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {1, 1, 1}};
        using STATE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        STATE state{}, baseline{};
        state.orientation[0] = 1;
        T commands[4];
        for(TI motor = 0; motor < 4; motor++){
            commands[motor] = dynamics.action_limit.min;
        }
        multirotor_dynamics(device, parameters, state, commands, baseline, l2f::actuation::Rotors{});
        for(TI motor = 0; motor < 4; motor++){
            SCOPED_TRACE(static_cast<int>(MODEL));
            SCOPED_TRACE(motor);
            const T minimum = dynamics.action_limit.min, maximum = dynamics.action_limit.max;
            const auto& coefficients = dynamics.rotor_thrust_coefficients[motor];
            const T thrust = coefficients[1] * (maximum - minimum) + coefficients[2] * (maximum * maximum - minimum * minimum);
            const T torque[3] = {
                signs[motor][1] * std::abs(dynamics.rotor_positions[motor][1]) * thrust,
                -signs[motor][0] * std::abs(dynamics.rotor_positions[motor][0]) * thrust,
                signs[motor][2] * dynamics.rotor_torque_constants[motor] * thrust
            };
            commands[motor] = maximum;
            STATE derivative{};
            multirotor_dynamics(device, parameters, state, commands, derivative, l2f::actuation::Rotors{});
            commands[motor] = minimum;
            for(TI axis = 0; axis < 3; axis++){
                const T linear = axis == 2 ? thrust / dynamics.mass : 0;
                EXPECT_NEAR(derivative.linear_velocity[axis] - baseline.linear_velocity[axis], linear, 1e-10 * (1 + std::abs(linear)));
                T angular = 0;
                for(TI component = 0; component < 3; component++){
                    angular += dynamics.J_inv[axis][component] * torque[component];
                }
                EXPECT_NEAR(derivative.angular_velocity[axis] - baseline.angular_velocity[axis], angular, 1e-10 * (1 + std::abs(angular)));
            }
        }
    }
}

TEST(L2F_MOTOR_ALLOCATION, REGISTRY_AND_SINGLE_MOTOR_WRENCHES){
    check_allocation<REGISTRY::crazyflie>();
    check_allocation<REGISTRY::crazyflie_openmv>();
    check_allocation<REGISTRY::mrs>();
    check_allocation<REGISTRY::x500>();
    check_allocation<REGISTRY::x500_real>();
    check_allocation<REGISTRY::x500_sim>();
    check_allocation<REGISTRY::arpl>();
    check_allocation<REGISTRY::fs_base>();
    check_allocation<REGISTRY::flightmare>();
    check_allocation<REGISTRY::soft>();
    check_allocation<REGISTRY::soft_rigid>();
    check_allocation<REGISTRY::crazyflie_brushless>();
    check_allocation<REGISTRY::savagebee_pusher>();
}

TEST(L2F_MOTOR_ALLOCATION, REJECTS_NONCANONICAL_LAYOUTS){
    using rlt::is_canonical_quadrotor;
    DEVICE device;
    const auto nominal = l2f::parameters::dynamics::registry<REGISTRY::crazyflie, SPEC>;
    auto dynamics = nominal;
    dynamics.rotor_positions[1][1] *= -1;
    EXPECT_FALSE(is_canonical_quadrotor(device, dynamics));
    dynamics = nominal;
    dynamics.rotor_torque_directions[0][2] = 1;
    EXPECT_FALSE(is_canonical_quadrotor(device, dynamics));
    dynamics = nominal;
    dynamics.rotor_thrust_directions[0][0] = T(0.1);
    EXPECT_FALSE(is_canonical_quadrotor(device, dynamics));
    dynamics = nominal;
    dynamics.rotor_torque_constants[0] *= -1;
    EXPECT_FALSE(is_canonical_quadrotor(device, dynamics));
    EXPECT_FALSE(is_canonical_quadrotor(device, l2f::parameters::Dynamics<T, TI, 6>{}));
}
