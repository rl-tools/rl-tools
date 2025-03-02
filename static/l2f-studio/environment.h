#include <rl_tools/operations/wasm32.h>
#include <rl_tools/rl/environments/l2f/operations_generic.h>

#include <rl_tools/rl/environments/l2f/multirotor.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultWASM32;
using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;

using T = float;
using TI = typename DEVICE::index_t;


namespace builder{
    using namespace rl_tools::rl::environments::l2f;
    static constexpr auto MODEL = parameters::dynamics::REGISTRY::crazyflie;
    constexpr static auto MODEL_NAME = parameters::dynamics::registry_name<MODEL>;
    using REWARD_FUNCTION = parameters::reward_functions::Squared<T>;
    static constexpr REWARD_FUNCTION reward_function = {
        false, // non-negative
        01.00, // scale
        01.50, // constant
        00.00, // termination penalty
        01.00, // position
        00.10, // orientation
        00.00, // linear_velocity
        00.00, // angular_velocity
        00.00, // linear_acceleration
        00.00, // angular_acceleration
        00.00, // action
        01.00, // d_action
    };

    using PARAMETERS_SPEC = ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = ParametersDomainRandomization<ParametersSpecification<T, TI, ParametersDisturbances<ParametersSpecification<T, TI, ParametersBase<PARAMETERS_SPEC>>>>>;

    static constexpr typename PARAMETERS_TYPE::Dynamics dynamics = parameters::dynamics::registry<MODEL, PARAMETERS_SPEC>;
    static constexpr TI SIMULATION_FREQUENCY = 100;
    static constexpr typename PARAMETERS_TYPE::Integration integration = {
        1.0/((T)SIMULATION_FREQUENCY) // integration dt
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = parameters::init::init_90_deg<PARAMETERS_SPEC>;
    static constexpr typename PARAMETERS_TYPE::MDP::ObservationNoise observation_noise = {
        0.00,// position
        0.00, // orientation
        0.00, // linear_velocity
        0.00, // angular_velocity
        0.00, // imu acceleration
    };
    static constexpr typename PARAMETERS_TYPE::MDP::ActionNoise action_noise = {
        0, // std of additive gaussian noise onto the normalized action (-1, 1)
    };
    static constexpr typename PARAMETERS_TYPE::MDP::Termination termination = {
        true,  // enable
        1,     // position
        2,    // linear velocity
        35,    // angular velocity
        10000, // position integral
        50000, // orientation integral
    };
    static constexpr typename PARAMETERS_TYPE::MDP mdp = {
        init,
        reward_function,
        observation_noise,
        action_noise,
        termination
    };
    static constexpr typename PARAMETERS_TYPE::DomainRandomization domain_randomization = {
        0.0, // thrust_to_weight_min;
        0.0, // thrust_to_weight_max;
        0.0, // thrust_to_weight_by_torque_to_inertia_min;
        0.0, // thrust_to_weight_by_torque_to_inertia_max;
        0.0, // mass_min;
        0.0, // mass_max;
        0.0, // mass_size_deviation;
        0.0, // motor_time_constant;
        0.0  // rotor_torque_constant;
    };
    static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {
        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0}, // random_force;
        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
    };
    static constexpr PARAMETERS_TYPE nominal_parameters = {
        {
            {
                dynamics,
                integration,
                mdp
            }, // Base
            disturbances
        }, // Disturbances
        domain_randomization
    }; // DomainRandomization

    struct ENVIRONMENT_STATIC_PARAMETERS{
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 16;
        static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>;
        using STATE_TYPE = StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>;
        using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>
        >>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto PARAMETER_VALUES = nominal_parameters;
        static constexpr TI N_DYNAMICS_VALUES = 1;
        static constexpr typename PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
            parameters::dynamics::registry<parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>
        };
        static constexpr T STATE_LIMIT_POSITION = 1000;
        static constexpr T STATE_LIMIT_VELOCITY = 100;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY = 100;
    };
}

using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::MultiTaskSpecification<T, TI, builder::ENVIRONMENT_STATIC_PARAMETERS>;
using ENVIRONMENT = rl_tools::rl::environments::MultirotorMultiTask<ENVIRONMENT_SPEC>;
