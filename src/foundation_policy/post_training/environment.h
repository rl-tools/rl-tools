#include "../pre_training/environment.h"

namespace builder{
    using namespace rl_tools;
    using namespace rl_tools::rl::environments::l2f;
    template <typename DEVICE, typename T, typename TI, typename OPTIONS>
    struct ENVIRONMENT_FACTORY_POST_TRAINING{

        using BASE_ENV = ENVIRONMENT_FACTORY<DEVICE, T, TI, OPTIONS>;

        using PARAMETERS_SPEC = rl_tools::rl::environments::l2f::ParametersBaseSpecification<T, TI, 4, typename BASE_ENV::REWARD_FUNCTION>;
        using PARAMETERS_TYPE = rl_tools::rl::environments::l2f::ParametersDisturbances<T, TI, rl_tools::rl::environments::l2f::ParametersBase<PARAMETERS_SPEC>>;
        static constexpr PARAMETERS_TYPE nominal_parameters = {
            {
                BASE_ENV::dynamics,
                BASE_ENV::integration,
                BASE_ENV::mdp,
                BASE_ENV::domain_randomization
            },
            BASE_ENV::disturbances
        };

        struct ENVIRONMENT_STATIC_PARAMETERS{
            static constexpr TI N_SUBSTEPS = 1;
            static constexpr TI ACTION_HISTORY_LENGTH = OPTIONS::SEQUENTIAL_MODEL ? 1 : 16;
            static constexpr TI EPISODE_STEP_LIMIT = 5 * BASE_ENV::SIMULATION_FREQUENCY;
            static constexpr TI CLOSED_FORM = false;
            static constexpr bool RANDOMIZE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES;
            static constexpr bool RANDOMIZE_MOTOR_MAPPING = OPTIONS::RANDOMIZE_MOTOR_MAPPING;
            static constexpr bool OBSERVE_THRUST_CURVES = OPTIONS::RANDOMIZE_THRUST_CURVES && OPTIONS::OBSERVE_THRASH_MARKOV;
            static constexpr bool OBSERVE_MOTOR_POSITIONS = OPTIONS::RANDOMIZE_MOTOR_MAPPING && OPTIONS::OBSERVE_THRASH_MARKOV;
            using STATE_BASE = StateLastAction<T, TI, StateBase<T, TI>>;
            using STATE_TYPE_MOTOR_DELAY = StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, StateRandomForce<T, TI, STATE_BASE>>;
            using STATE_TYPE_NO_MOTOR_DELAY = StateRandomForce<T, TI, STATE_BASE>;
            using STATE_TYPE = rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, STATE_TYPE_MOTOR_DELAY, STATE_TYPE_NO_MOTOR_DELAY>;
            using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                    observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                    observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_THRUST_CURVES, observation::ParametersThrustCurves<observation::ParametersThrustCurvesSpecification<T, TI, BASE_ENV::PARAMETERS_TYPE::N>>,
                    observation::Multiplex<observation::MultiplexSpecification<TI, OBSERVE_MOTOR_POSITIONS, observation::ParametersMotorPosition<observation::ParametersMotorPositionSpecification<T, TI, BASE_ENV::PARAMETERS_TYPE::N>>,
                    rl_tools::utils::typing::conditional_t<OPTIONS::MOTOR_DELAY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>, observation::LastComponent<TI>>
                    // observation::ParametersMass<observation::ParametersMassSpecification<T, TI
            >>>>>>>>>>>>;
            using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
            static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            using PARAMETERS = typename BASE_ENV::PARAMETERS_TYPE;
            static constexpr auto PARAMETER_VALUES = BASE_ENV::nominal_parameters;
            static constexpr TI N_DYNAMICS_VALUES = 1;
            static constexpr typename BASE_ENV::PARAMETERS_TYPE::Dynamics DYNAMICS_VALUES[N_DYNAMICS_VALUES] = {
                rl_tools::rl::environments::l2f::parameters::dynamics::registry<rl_tools::rl::environments::l2f::parameters::dynamics::REGISTRY::crazyflie, BASE_ENV::PARAMETERS_SPEC>
            };
        };

        using ENVIRONMENT_SPEC = rl_tools::rl::environments::l2f::MultiTaskSpecification<T, TI, ENVIRONMENT_STATIC_PARAMETERS, OPTIONS::SAMPLE_INITIAL_PARAMETERS>;
        using ENVIRONMENT = rl_tools::rl::environments::MultirotorMultiTask<ENVIRONMENT_SPEC>;
        static_assert(rl::environments::PREVENT_DEFAULT_GET_UI<ENVIRONMENT>::value);
    };
}
