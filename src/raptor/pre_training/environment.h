#include <rl_tools/rl/environments/l2f/operations_multitask_generic_forward.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/operations_multitask_generic.h>

// #include <rl_tools/rl/environments/l2f/persist_code.h>


#include <rl_tools/utils/generic/typing.h>

namespace builder{
    using namespace rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY{

        using BASE_FACTORY = rl_tools::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI>;

        static constexpr auto MODEL = BASE_FACTORY::MODEL;
        constexpr static auto MODEL_NAME = rl_tools::rl::environments::l2f::parameters::dynamics::registry_name<MODEL>;

        using REWARD_FUNCTION = typename BASE_FACTORY::REWARD_FUNCTION;
        using PARAMETERS_SPEC = typename BASE_FACTORY::PARAMETERS_SPEC;
        using PARAMETERS_TYPE = typename BASE_FACTORY::PARAMETERS_TYPE;

        static constexpr TI SIMULATION_FREQUENCY = BASE_FACTORY::SIMULATION_FREQUENCY;

        static constexpr PARAMETERS_TYPE nominal_parameters = BASE_FACTORY::nominal_parameters;

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr auto ACTION_INTERFACE = parameters::ActionInterface::DIRECT_MOTOR;
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = 1;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr TI ANGULAR_VELOCITY_DELAY = 0; // one step at 100hz = 10ms ~ delay from IMU to input to the policy: 1.3ms time constant of the IIR in the IMU (bw ~110Hz) + synchronization delay (2ms) + (negligible SPI transfer latency due to it being interrupt-based) + 1ms sensor.c RTOS loop @ 1khz + 2ms for the RLtools loop
            using STATE_BASE = StateAngularVelocityDelay<StateAngularVelocityDelaySpecification<T, TI, ANGULAR_VELOCITY_DELAY, StateLastAction<StateSpecification<T, TI, StateBase<StateSpecification<T, TI>>>>>>;
            using STATE_TYPE_MOTOR_DELAY = StateTrajectory<StateSpecification<T, TI, StateRotorsHistory<StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<StateSpecification<T, TI, STATE_BASE>>>>>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<StateSpecification<T, TI, STATE_BASE>>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::TrajectoryTrackingPosition<observation::TrajectoryTrackingPositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::TrajectoryTrackingLinearVelocity<observation::TrajectoryTrackingLinearVelocitySpecification<T, TI,
                    observation::AngularVelocityDelayed<observation::AngularVelocityDelayedSpecification<T, TI, ANGULAR_VELOCITY_DELAY,
                    // observation::RandomForce<observation::RandomForceSpecification<T, TI,
                    observation::ActionHistory<observation::ActionHistorySpecification<T, TI, 1, // one-step action history to Markovify the d_action regularization
                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>>
            >>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = nominal_parameters;
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

        using ENVIRONMENT_SPEC = Specification<T, TI, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        // static_assert(rl::environments::PREVENT_DEFAULT_GET_UI<ENVIRONMENT>::value);
    };
}
