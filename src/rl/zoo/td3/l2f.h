
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/abs_exp.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/squared.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/absolute.h>
#include <rl_tools/rl/environments/l2f/parameters/reward_functions/default.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/crazy_flie.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/race.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_sim.h>
#include <rl_tools/rl/environments/l2f/parameters/dynamics/x500_real.h>
#include <rl_tools/rl/environments/l2f/parameters/init/default.h>
#include <rl_tools/rl/environments/l2f/parameters/termination/default.h>


#include <rl_tools/utils/generic/typing.h>

namespace rl_tools::rl::zoo::td3::l2f{
    namespace rlt = rl_tools;
    struct DefaultAblationSpec{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = true;
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
        static constexpr bool INIT_NORMAL = true;
        static constexpr bool DOMAIN_RANDOMIZATION = true;
    };
    namespace builder {
        namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER::rl_tools;
        using namespace rlt::rl::environments::multirotor;
        template<typename T, typename TI>
        struct ENVIRONMENT_BUILDER{
            static constexpr auto reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_absolute<T>;
            using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
            using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

            constexpr static auto MODEL = rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY::crazyflie;
//            constexpr static auto MODEL = rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY::fs_base;
            using PARAMETERS_SPEC = rl_tools::rl::environments::multirotor::ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION, rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY, MODEL>;
            using PARAMETERS_TYPE = rl_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, rl_tools::rl::environments::multirotor::ParametersBase<PARAMETERS_SPEC>>;

            static constexpr auto init_params = rl_tools::rl::environments::multirotor::parameters::init::orientation_biggest_angle<PARAMETERS_SPEC>;

            constexpr static auto MODEL_NAME = rl_tools::rl::environments::multirotor::parameters::dynamics::registry_name<PARAMETERS_SPEC>;
            constexpr static auto dynamics_parameters = rl_tools::rl::environments::multirotor::parameters::dynamics::registry<PARAMETERS_SPEC>; //rl_tools::rl::environments::multirotor::parameters::dynamics::x500::real<PARAMETERS_SPEC>;
            static constexpr PARAMETERS_TYPE parameters = {
                    dynamics_parameters,
                    {0.01}, // integration dt
                    {
                            init_params,
                            reward_function,
                            {   // Observation noise
                                    0.05, // position
                                    0.001, // orientation
                                    0.1, // linear_velocity
                                    0.2, // angular_velocity
                            },
                            {   // Action noise
                                    0, // std of additive gaussian noise onto the normalized action (-1, 1)
                            },
                            rl_tools::rl::environments::multirotor::parameters::termination::fast_learning<PARAMETERS_SPEC>
                    },
                    { // domain  randomization
                        0 //thrust_coefficients
                    },
                    typename PARAMETERS_TYPE::Disturbances{
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 1.0 * 9.81 / 20}, // random_force;
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 1.0 * 9.81 / 1000} // random_torque;
                    }

            };

//            using PARAMETERS = typename rl_tools::utils::typing::remove_cv_t<decltype(parameters)>;

            struct ENVIRONMENT_STATIC_PARAMETERS{
                static constexpr TI ACTION_HISTORY_LENGTH = 32;
//                using STATE_BASE = StatePoseErrorIntegral<T, TI, StateBase<T, TI>>;
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
            };

            using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS_TYPE, ENVIRONMENT_STATIC_PARAMETERS>;
            using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        };
    }
    template <typename DEVICE, typename T, typename TI, typename RNG>
    struct LearningToFly{
        using ENVIRONMENT = typename builder::ENVIRONMENT_BUILDER<T, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
            static constexpr TI STEP_LIMIT = 300000;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::td3::loop::core::ConfigApproximatorsSequential>;
    };
}
