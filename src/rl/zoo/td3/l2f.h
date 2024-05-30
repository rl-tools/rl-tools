#ifndef RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H
#define RL_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H

#include <learning_to_fly/simulator/parameters/reward_functions/abs_exp.h>
#include <learning_to_fly/simulator/parameters/reward_functions/squared.h>
#include <learning_to_fly/simulator/parameters/reward_functions/absolute.h>
#include <learning_to_fly/simulator/parameters/reward_functions/default.h>
#include <learning_to_fly/simulator/parameters/dynamics/crazy_flie.h>
#include <learning_to_fly/simulator/parameters/dynamics/race.h>
#include <learning_to_fly/simulator/parameters/dynamics/x500_sim.h>
#include <learning_to_fly/simulator/parameters/dynamics/x500_real.h>
#include <learning_to_fly/simulator/parameters/init/default.h>
#include <learning_to_fly/simulator/parameters/termination/default.h>

#include <rl_tools/utils/generic/typing.h>

namespace parameters{
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
        template<typename T, typename TI, typename T_ABLATION_SPEC>
        struct environment {
            using ABLATION_SPEC = T_ABLATION_SPEC;
#ifdef LEARNING_TO_FLY_SQUARED_REWARD_FUNCTION
            static constexpr auto initial_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
            static constexpr auto target_reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target<T>;
            static constexpr auto reward_function = ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? initial_reward_function : target_reward_function;
#else
            static constexpr auto reward_function = rl_tools::rl::environments::multirotor::parameters::reward_functions::reward_absolute<T>;
#endif


            using REWARD_FUNCTION_CONST = typename rl_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
            using REWARD_FUNCTION = typename rl_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

            constexpr static auto MODEL = rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY::crazyflie;
//            constexpr static auto MODEL = rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY::fs_base;
            using PARAMETERS_SPEC = rl_tools::rl::environments::multirotor::ParametersBaseSpecification<T, TI, 4, REWARD_FUNCTION, rl_tools::rl::environments::multirotor::parameters::dynamics::REGISTRY, MODEL>;
            using PARAMETERS_TYPE = rl_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, rl_tools::rl::environments::multirotor::ParametersBase<PARAMETERS_SPEC>>;

            static_assert(ABLATION_SPEC::INIT_NORMAL);
            static constexpr auto init_params = ABLATION_SPEC::INIT_NORMAL ?
                                                rl_tools::rl::environments::multirotor::parameters::init::orientation_biggest_angle<PARAMETERS_SPEC>
                                                                           :
                                                rl_tools::rl::environments::multirotor::parameters::init::all_positions<PARAMETERS_SPEC>;

            constexpr static auto MODEL_NAME = rl_tools::rl::environments::multirotor::parameters::dynamics::registry_name<PARAMETERS_SPEC>;
            constexpr static auto dynamics_parameters = rl_tools::rl::environments::multirotor::parameters::dynamics::registry<PARAMETERS_SPEC>; //rl_tools::rl::environments::multirotor::parameters::dynamics::x500::real<PARAMETERS_SPEC>;
            static constexpr PARAMETERS_TYPE parameters = {
                    dynamics_parameters,
                    {0.01}, // integration dt
                    {
                            init_params,
                            reward_function,
                            {   // Observation noise
                                    0.05 * ABLATION_SPEC::OBSERVATION_NOISE, // position
                                    0.001 * ABLATION_SPEC::OBSERVATION_NOISE, // orientation
                                    0.1 * ABLATION_SPEC::OBSERVATION_NOISE, // linear_velocity
                                    0.2 * ABLATION_SPEC::OBSERVATION_NOISE, // angular_velocity
                            },
                            {   // Action noise
                                    0, // std of additive gaussian noise onto the normalized action (-1, 1)
                            },
                            rl_tools::rl::environments::multirotor::parameters::termination::fast_learning<PARAMETERS_SPEC>
                    },
                    { // domain  randomization
                        0 * ABLATION_SPEC::DOMAIN_RANDOMIZATION //thrust_coefficients
                    },
                    typename PARAMETERS_TYPE::Disturbances{
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 1.0 * 9.81 / 20 * ABLATION_SPEC::DISTURBANCE}, // random_force;
                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 1.0 * 9.81 / 1000 * ABLATION_SPEC::DISTURBANCE} // random_torque;
                    }

            };

//            using PARAMETERS = typename rl_tools::utils::typing::remove_cv_t<decltype(parameters)>;

            struct ENVIRONMENT_STATIC_PARAMETERS{
                static constexpr TI ACTION_HISTORY_LENGTH = 32;
//                using STATE_BASE = StatePoseErrorIntegral<T, TI, StateBase<T, TI>>;
                using STATE_BASE = StateBase<T, TI>;
                using STATE_TYPE = rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                    rlt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY,
                        StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, STATE_BASE>, STATE_BASE>>,
                        StateRotors<T, TI, rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, STATE_BASE>, STATE_BASE>>>,
                    rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, STATE_BASE>, STATE_BASE>>;
                using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                        observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                        observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                                rlt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>, observation::LastComponent<TI>>>>>>>>>>;
                using OBSERVATION_TYPE_PRIVILEGED = rlt::utils::typing::conditional_t<ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC,
//                    observation::PoseIntegral<observation::PoseIntegralSpecification<T, TI,
                        observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                                observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                        rlt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE,
                                            observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                                >
                                            >>,
                                            rlt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                            >
                                        >
                                    >>
                                >>
                            >>
//                        >>
                    >>,
                    observation::NONE<TI>
                >;
                static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
            };

            using ENVIRONMENT_SPEC = rlt::rl::environments::multirotor::Specification<T, TI, PARAMETERS_TYPE, ENVIRONMENT_STATIC_PARAMETERS>;
            using ENVIRONMENT = rlt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
        };
    }
    template<typename T, typename TI, typename ABLATION_SPEC>
    using environment = builder::environment<T, TI, ABLATION_SPEC>;
}

#endif
