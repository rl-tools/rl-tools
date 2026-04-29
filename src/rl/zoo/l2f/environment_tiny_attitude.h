#include <rl_tools/version.h>
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_TINY_ATTITUDE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ZOO_L2F_ENVIRONMENT_TINY_ATTITUDE_H

#include "reward_attitude.h"
#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/squared/operations_generic.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/default.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/crazyflie.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/arpl.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_sim.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_real.h>
#include <rl_tools/rl/environments/l2f/parameters/init/default.h>
#include <rl_tools/rl/environments/l2f/parameters/termination/default.h>

#include <rl_tools/rl/environments/l2f/persist_code.h>


#include "environment.h"
#include <rl_tools/utils/generic/typing.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::l2f{
    namespace rlt = rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename TYPE_POLICY, typename TI>
    struct ENVIRONMENT_TINY_ATTITUDE_FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        static constexpr TI SIMULATION_FREQUENCY = 50;
        static constexpr TI EPISODE_LENGTH_S = 5;
        using ENVIRONMENT_FACTORY_BASE = ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI, EPISODE_LENGTH_S, SIMULATION_FREQUENCY>;
        using REWARD_FUNCTION = rl_tools::rl::environments::l2f::parameters::reward_functions::AttitudeSquared<T>;
        // Rebuild the PARAMETERS wrapping with our AttitudeSquared reward. All other
        // nested types (Dynamics, Integration, Termination, ...) resolve to
        // reward-function-independent concrete types, so the values from the base
        // factory remain assignable into this spec.
        using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER, REWARD_FUNCTION>;
        using PARAMETERS_TYPE = ParametersTrajectory<ParametersTrajectorySpecification<T, TI, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_LENGTH, ENVIRONMENT_FACTORY_BASE::TRAJECTORY_DT, ParametersDomainRandomization<ParametersDomainRandomizationSpecification<T, TI, DefaultParametersDomainRandomizationOptions, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>>>;

        static constexpr auto MODEL = rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;
        static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = [](){
            auto p = rl_tools::rl::environments::l2f::parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
            p.rotor_time_constants_rising[0] = 0.072;
            p.rotor_time_constants_rising[1] = 0.072;
            p.rotor_time_constants_rising[2] = 0.072;
            p.rotor_time_constants_rising[3] = 0.072;
            p.rotor_time_constants_falling[0] = 0.072;
            p.rotor_time_constants_falling[1] = 0.072;
            p.rotor_time_constants_falling[2] = 0.072;
            p.rotor_time_constants_falling[3] = 0.072;
            p.mass = 0.025;
            p.rotor_thrust_coefficients[0][0] = 0;
            p.rotor_thrust_coefficients[1][0] = 0;
            p.rotor_thrust_coefficients[2][0] = 0;
            p.rotor_thrust_coefficients[3][0] = 0;
            p.rotor_thrust_coefficients[0][1] = 0;
            p.rotor_thrust_coefficients[1][1] = 0;
            p.rotor_thrust_coefficients[2][1] = 0;
            p.rotor_thrust_coefficients[3][1] = 0;
            p.rotor_thrust_coefficients[0][2] = 0.1302;
            p.rotor_thrust_coefficients[1][2] = 0.1302;
            p.rotor_thrust_coefficients[2][2] = 0.1302;
            p.rotor_thrust_coefficients[3][2] = 0.1302;
            return p;
        }();

        static constexpr typename ParametersBase<PARAMETERS_SPEC>::MDP::Initialization init = {
                0.0, // guidance
                0,   // position
                1.5707963267948966 * 60.0/90.0,   // orientation (~60 deg)
                0,   // linear velocity
                1,   // angular velocity
                true,// relative rpm
                -1,  // min rpm
                +1,  // max rpm
        };
        static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
                true,  // enabled
                10,    // position (effectively disabled, state is clamped to STATE_LIMIT_POSITION)
                1.5707963267948966, // angle (90 deg)
                20,    // linear velocity (effectively disabled, state is clamped to STATE_LIMIT_VELOCITY)
                35,    // angular velocity (safety net)
                10000, // position integral
                50000, // orientation integral
        };
        // Tilt cost is 1 - R[2][2] — proper roll/pitch penalty with direct gradient
        // at any tilt, unlike Squared's orientation_cost which only penalizes yaw.
        static constexpr REWARD_FUNCTION reward_function = {
                false, // non-negative
                01.00, // scale
                01.00, // constant
                00.00, // termination penalty (unused, see reward_attitude.h)
                02.00, // tilt
                00.05, // angular_velocity
                00.05, // action
        };
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {
            init,
            reward_function,
            ENVIRONMENT_FACTORY_BASE::observation_noise,
            ENVIRONMENT_FACTORY_BASE::action_noise,
            termination
        };
        static constexpr typename PARAMETERS_TYPE::Integration integration = {
            1.0/((T)SIMULATION_FREQUENCY) // integration dt
        };

        static constexpr decltype(ENVIRONMENT_FACTORY_BASE::trajectory) trajectory = {};
        static constexpr typename PARAMETERS_TYPE::TrajectoryParameters trajectory_parameters = {
            rl::environments::l2f::parameters::trajectories::Type::LISSAJOUS,
            {
                rl::environments::l2f::parameters::trajectories::lissajous::Parameters<T>{
                    0, // A
                    0, // B
                    0, // C
                    1, // a
                    1, // b
                    1, // c
                    1, // interval
                    0  // ramp_duration
                }
            }
        };

        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                {
                    {
                        dynamics,
                        integration,
                        mdp
                    },
                    ENVIRONMENT_FACTORY_BASE::disturbances
                },
                ENVIRONMENT_FACTORY_BASE::domain_randomization
            },
            trajectory,
            trajectory_parameters
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 2;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT_FACTORY_BASE::EPISODE_STEP_LIMIT_OUTER;
            static constexpr TI CLOSED_FORM = false;
            using STATE_BASE = StateBase<StateSpecification<T, TI>>;
            using STATE_TYPE = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
            using OBSERVATION_TYPE = observation::OrientationBodyZ<observation::OrientationBodyZSpecification<T, TI,
                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                            observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = typename ENVIRONMENT_FACTORY_BASE::ENVIRONMENT_STATIC_PARAMETERS::OBSERVATION_TYPE_PRIVILEGED;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
            static constexpr T STATE_LIMIT_POSITION = 0;
            static constexpr T STATE_LIMIT_VELOCITY = 0;
            static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100000;
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
