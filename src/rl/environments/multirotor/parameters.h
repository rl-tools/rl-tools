#ifndef BACKPROP_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H
#define BACKPROP_TOOLS_SRC_RL_ENVIRONMENTS_MULTIROTOR_PARAMETERS_H

#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/absolute.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <backprop_tools/rl/environments/multirotor/parameters/init/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/termination/default.h>

#include <backprop_tools/utils/generic/typing.h>

namespace parameters{
    namespace bpt = backprop_tools;
    struct DefaultAblationSpec{
        static constexpr bool DISTURBANCE = true;
        static constexpr bool OBSERVATION_NOISE = true;
        static constexpr bool ASYMMETRIC_ACTOR_CRITIC = true;
        static constexpr bool ROTOR_DELAY = true;
        static constexpr bool ACTION_HISTORY = true;
        static constexpr bool ENABLE_CURRICULUM = true;
        static constexpr bool USE_INITIAL_REWARD_FUNCTION = true;
        static constexpr bool INIT_NORMAL = true;
    };
    namespace sim2real{
        namespace builder {
            namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER::backprop_tools;
            using namespace bpt::rl::environments::multirotor;
            template<typename T, typename TI, typename T_ABLATION_SPEC>
            struct environment {
                using ABLATION_SPEC = T_ABLATION_SPEC;
                //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold_4<T>;
                //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
                //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::sq_exp_position_action_only_3<T>;
                //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::sq_exp_reward_mm<T, TI>;
//                static constexpr auto reward_function = ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ?
//                                                        backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>
//                                                                                                   :
//                                                        backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_263<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_fast_learning<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_absolute_fast_learning<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_absolute_fast_learning_2<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::abs_exp_position_only<T>;
//                static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::abs_exp_position_orientation_lin_vel<T>;
                static constexpr auto initial_reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
                static constexpr auto target_reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque_curriculum_target<T>;
                static constexpr auto reward_function = ABLATION_SPEC::USE_INITIAL_REWARD_FUNCTION ? initial_reward_function : target_reward_function;


                using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
                using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

                using PARAMETERS_TYPE = backprop_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, backprop_tools::rl::environments::multirotor::ParametersBase<T, TI, 4, REWARD_FUNCTION>>;

                static_assert(ABLATION_SPEC::INIT_NORMAL);
                static constexpr auto init_params = ABLATION_SPEC::INIT_NORMAL ?
                                                    backprop_tools::rl::environments::multirotor::parameters::init::orientation_biggest_angle<T, TI, 4, REWARD_FUNCTION>
                                                                               :
//                                                    backprop_tools::rl::environments::multirotor::parameters::init::orientation_small_angle<T, TI, 4, REWARD_FUNCTION>;
                                                    backprop_tools::rl::environments::multirotor::parameters::init::all_positions<T, TI, 4, REWARD_FUNCTION>;

                static constexpr PARAMETERS_TYPE parameters = {
                        backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie_old_reduced_inertia<T, TI, REWARD_FUNCTION>,
                        {0.01}, // integration dt
                        {
                                //                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_orientation_only<T, TI, 4, REWARD_FUNCTION>,
                                init_params,
                                //                        backprop_tools::rl::environments::multirotor::parameters::init::orientation_all_around<T, TI, 4, REWARD_FUNCTION>,
                                //                        backprop_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
                                reward_function,
                                {   // Observation noise
                                        0.001 * ABLATION_SPEC::OBSERVATION_NOISE, // position
                                        0.001 * ABLATION_SPEC::OBSERVATION_NOISE, // orientation
                                        0.002 * ABLATION_SPEC::OBSERVATION_NOISE, // linear_velocity
                                        0.002 * ABLATION_SPEC::OBSERVATION_NOISE, // angular_velocity
                                },
                                {   // Action noise
                                        0, // std of additive gaussian noise onto the normalized action (-1, 1)
                                },
                                backprop_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                        },
                        typename PARAMETERS_TYPE::Disturbances{
                                typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 20 *
                                                                                              ABLATION_SPEC::DISTURBANCE}, // random_force;
                                //                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_force;
                                typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10000 *
                                                                                              ABLATION_SPEC::DISTURBANCE} // random_torque;
                        }

                };

                using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;

                struct ENVIRONMENT_STATIC_PARAMETERS{
                    static constexpr TI ACTION_HISTORY_LENGTH = 32;
                    using STATE_TYPE = bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                        bpt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY,
                            StateRotorsHistory<T, TI, ACTION_HISTORY_LENGTH, bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>,
                            StateRotors<T, TI, bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>>,
                        bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE, StateRandomForce<T, TI, StateBase<T, TI>>, StateBase<T, TI>>>;
                    using OBSERVATION_TYPE = observation::Position<observation::PositionSpecification<T, TI,
                            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecification<T, TI,
                                    observation::LinearVelocity<observation::LinearVelocitySpecification<T, TI,
                                            observation::AngularVelocity<observation::AngularVelocitySpecification<T, TI,
                                                    bpt::utils::typing::conditional_t<ABLATION_SPEC::ACTION_HISTORY, observation::ActionHistory<observation::ActionHistorySpecification<T, TI, ACTION_HISTORY_LENGTH>>, observation::LastComponent<TI>>>>>>>>>>;
                    using OBSERVATION_TYPE_PRIVILEGED = bpt::utils::typing::conditional_t<ABLATION_SPEC::ASYMMETRIC_ACTOR_CRITIC,
                        observation::Position<observation::PositionSpecificationPrivileged<T, TI,
                            observation::OrientationRotationMatrix<observation::OrientationRotationMatrixSpecificationPrivileged<T, TI,
                                observation::LinearVelocity<observation::LinearVelocitySpecificationPrivileged<T, TI,
                                    observation::AngularVelocity<observation::AngularVelocitySpecificationPrivileged<T, TI,
                                        bpt::utils::typing::conditional_t<ABLATION_SPEC::DISTURBANCE,
                                            observation::RandomForce<observation::RandomForceSpecification<T, TI,
                                                bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                                >
                                            >>,
                                            bpt::utils::typing::conditional_t<ABLATION_SPEC::ROTOR_DELAY,
                                                    observation::RotorSpeeds<observation::RotorSpeedsSpecification<T, TI>>,
                                                    observation::LastComponent<TI>
                                            >
                                        >
                                    >>
                                >>
                            >>
                        >>,
                        observation::NONE<TI>
                    >;
                    static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
                };

                using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
                using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
            };
        }
        template<typename T, typename TI, typename ABLATION_SPEC>
        using environment = builder::environment<T, TI, ABLATION_SPEC>;
    }

//    namespace fast_learning{
//        template<typename T, typename TI>
//        struct environment{
//            static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
//            //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
//            //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_2<T>;
//            //        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_4<T>;
//            using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
//            using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;
//
//            using PARAMETERS_TYPE = backprop_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, 4, REWARD_FUNCTION>;
//            static constexpr PARAMETERS_TYPE parameters = {
//                    backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie_old<T, TI, REWARD_FUNCTION>,
//                    {0.01}, // integration dt
//                    {
//                            backprop_tools::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
//                            reward_function,
//                            {   // Observation noise
//                                    0, // position
//                                    0, // orientation
//                                    0, // linear_velocity
//                                    0, // angular_velocity
//                            },
//                            {   // Action noise
//                                    0, // std of additive gaussian noise onto the normalized action (-1, 1)
//                            },
//                            //                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_simplified<T, TI, 4, REWARD_FUNCTION>,
//                            //                        backprop_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
//                            backprop_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
//                    },
//                    typename PARAMETERS_TYPE::Disturbances{
//                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10}, // random_force;
//                            typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
//                    }
//            };
//
//            using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;
//
//            struct ENVIRONMENT_STATIC_PARAMETERS: bpt::rl::environments::multirotor::StaticParametersDefault<TI>{
//                static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//                static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//                static constexpr bpt::rl::environments::multirotor::StateType STATE_TYPE = bpt::rl::environments::multirotor::StateType::Base;
//                static constexpr bpt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = bpt::rl::environments::multirotor::ObservationType::Normal;
//            };
//
//            using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
//            using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
//        };
//    }
}

#endif
