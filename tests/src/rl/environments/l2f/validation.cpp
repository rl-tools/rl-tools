#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>

#include <nlohmann/json.hpp>
#include <fstream>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = double;
using TI = typename DEVICE::index_t;


constexpr T EPSILON = 1e-6;

namespace static_parameter_builder{
    // to prevent spamming the global namespace
    using namespace rl_tools::rl::environments::l2f;
    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        using STATE_BASE = StateBase<T, TI>;
        using STATE_TYPE = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, StateRandomForce<T, TI, STATE_BASE>>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                observation::RandomForce<observation::RandomForceSpecification<T, TI,
                observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>
        >
        >
        >>
        >>
        >>
        >>;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETER_FACTORY = parameters::DefaultParameters<T, TI>;
        static constexpr auto PARAMETER_VALUES = PARAMETER_FACTORY::parameters;
        using PARAMETERS = typename PARAMETER_FACTORY::PARAMETERS_TYPE;
    };
}

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, static_parameter_builder::ENVIRONMENT_STATIC_PARAMETERS>;
using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;


template <typename DEVICE>
ENVIRONMENT::State parse_state(DEVICE& device, ENVIRONMENT& env, ENVIRONMENT::State& base_state, ENVIRONMENT::Parameters& parameters, std::vector<T> step_state){
    ENVIRONMENT::State state = base_state;
    rlt::initial_state(device, env, parameters, state);
    state.position[0] = step_state[0];
    state.position[1] = step_state[1];
    state.position[2] = step_state[2];
    state.orientation[0] = step_state[3];
    state.orientation[1] = step_state[4];
    state.orientation[2] = step_state[5];
    state.orientation[3] = step_state[6];
    state.linear_velocity[0] = step_state[7];
    state.linear_velocity[1] = step_state[8];
    state.linear_velocity[2] = step_state[9];
    state.angular_velocity[0] = step_state[10];
    state.angular_velocity[1] = step_state[11];
    state.angular_velocity[2] = step_state[12];
    state.rpm[0] = step_state[13];
    state.rpm[1] = step_state[14];
    state.rpm[2] = step_state[15];
    state.rpm[3] = step_state[16];
    return state;
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_L2F, VALIDATION) {
    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM{}, 0);
    std::string path = "/Users/jonas/git/flightmare/quad_dynamics.json";
    std::ifstream ifs(path);
    nlohmann::json j = nlohmann::json::parse(ifs);
    // print
//    std::cout << j.dump(4) << std::endl;

    ENVIRONMENT env;
    ENVIRONMENT::State state, next_state;
    ENVIRONMENT::Parameters parameters;
    rlt::malloc(device, env);
    rlt::initial_parameters(device, env, parameters);
    rlt::initial_state(device, env, parameters, state);

    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    parameters.dynamics.mass = j["dynamics"]["mass"];
    T radius = j["dynamics"]["radius"];
    parameters.dynamics.J[0][0] = j["dynamics"]["J"]["data"][0];
    parameters.dynamics.J[1][1] = j["dynamics"]["J"]["data"][4];
    parameters.dynamics.J[2][2] = j["dynamics"]["J"]["data"][8];
    parameters.dynamics.J_inv[0][0] = 1.0 / parameters.dynamics.J[0][0];
    parameters.dynamics.J_inv[1][1] = 1.0 / parameters.dynamics.J[1][1];
    parameters.dynamics.J_inv[2][2] = 1.0 / parameters.dynamics.J[2][2];
    parameters.dynamics.rotor_thrust_coefficients[0] = j["dynamics"]["thrust_curve"]["data"][0];
    parameters.dynamics.rotor_thrust_coefficients[1] = j["dynamics"]["thrust_curve"]["data"][1];
    parameters.dynamics.rotor_thrust_coefficients[2] = j["dynamics"]["thrust_curve"]["data"][2];
    parameters.dynamics.rotor_torque_constant = j["dynamics"]["torque_constant"];
    parameters.dynamics.action_limit.min = 0;
    parameters.dynamics.action_limit.max = 1;
    parameters.dynamics.motor_time_constant = j["dynamics"]["motor_time_constant"];

    for(TI trajectory_i = 0; trajectory_i < j["trajectories"].size(); trajectory_i++){
        auto trajectory = j["trajectories"][trajectory_i];
        std::vector<std::vector<T>> rpm_setpoints = trajectory["rpm_setpoints"];
        std::vector<std::vector<T>> states = trajectory["states"];
        for(TI step_i = 0; step_i < states.size(); step_i++){
            std::vector<T> step_state = states[step_i];
            ENVIRONMENT::State target_state = parse_state(device, env, state, parameters, step_state);
            if(step_i == 0){
                state = target_state;
            }
            rlt::set(action, 0, 0, rpm_setpoints[step_i][0]);
            rlt::set(action, 0, 1, rpm_setpoints[step_i][1]);
            rlt::set(action, 0, 2, rpm_setpoints[step_i][2]);
            rlt::set(action, 0, 3, rpm_setpoints[step_i][3]);
            rlt::step(device, env, parameters, state, action, next_state, rng);
            if(step_i < states.size()-1){
                ENVIRONMENT::State target_next_state = parse_state(device, env, state, parameters, states[step_i+1]);
                std::cout << "step i: " << step_i << std::endl;
//                ASSERT_NEAR(target_next_state.position[0], next_state.position[0], EPSILON);
//                ASSERT_NEAR(target_next_state.position[1], next_state.position[1], EPSILON);
//                ASSERT_NEAR(target_next_state.position[2], next_state.position[2], EPSILON);
//                ASSERT_NEAR(target_next_state.orientation[0], next_state.orientation[0], EPSILON);
//                ASSERT_NEAR(target_next_state.orientation[1], next_state.orientation[1], EPSILON);
//                ASSERT_NEAR(target_next_state.orientation[2], next_state.orientation[2], EPSILON);
//                ASSERT_NEAR(target_next_state.orientation[3], next_state.orientation[3], EPSILON);
//                ASSERT_NEAR(target_next_state.linear_velocity[0], next_state.linear_velocity[0], EPSILON);
//                ASSERT_NEAR(target_next_state.linear_velocity[1], next_state.linear_velocity[1], EPSILON);
//                ASSERT_NEAR(target_next_state.linear_velocity[2], next_state.linear_velocity[2], EPSILON);

            }
        }

    }



    rlt::step(device, env, parameters, state, action, next_state, rng);
}