#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <backprop_tools/rl/environments/multirotor/parameters/init/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/termination/default.h>

namespace parameters_sim2real_old{
    namespace bpt = backprop_tools;
    template<typename T, typename TI>
    struct environment{
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold_4<T>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::sq_exp_position_action_only_3<T>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::sq_exp_reward_mm<T, TI>;
        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_position_only_torque<T>;
        using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_TYPE = backprop_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, 4, REWARD_FUNCTION>;
        static constexpr PARAMETERS_TYPE parameters = {
                backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie_old_reduced_inertia<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
//                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_orientation_only<T, TI, 4, REWARD_FUNCTION>,
                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_2<T, TI, 4, REWARD_FUNCTION>,
//                        backprop_tools::rl::environments::multirotor::parameters::init::orientation_all_around<T, TI, 4, REWARD_FUNCTION>,
//                        backprop_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                        {   // Observation noise
                            0.0, // position
                            0.0, // orientation
                            0.0, // linear_velocity
                            0.0, // angular_velocity
//                            0.001, // position
//                            0.001, // orientation
//                            0.002, // linear_velocity
//                            0.002, // angular_velocity
                        },
                        {   // Action noise
                            0, // std of additive gaussian noise onto the normalized action (-1, 1)
                        },
                        backprop_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                },
                typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 20}, // random_force;
//                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10000} // random_torque;
                }

        };

        using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;

        struct ENVIRONMENT_STATIC_PARAMETERS: bpt::rl::environments::multirotor::StaticParametersDefault<TI>{
            static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
            static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
            static constexpr bpt::rl::environments::multirotor::LatentStateType LATENT_STATE_TYPE = bpt::rl::environments::multirotor::LatentStateType::RandomForce;
#if defined(ENABLE_MULTI_CONFIG)
            static constexpr bpt::rl::environments::multirotor::StateType STATE_TYPE = JOB_ID % 2 == 0 ? bpt::rl::environments::multirotor::StateType::BaseRotorsHistory : bpt::rl::environments::multirotor::StateType::BaseRotors;
            static constexpr TI ACTION_HISTORY_LENGTH = 48;
#else
            static constexpr bpt::rl::environments::multirotor::StateType STATE_TYPE = bpt::rl::environments::multirotor::StateType::BaseRotorsHistory;
            static constexpr TI ACTION_HISTORY_LENGTH = 32;
#endif
            static constexpr bpt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = bpt::rl::environments::multirotor::ObservationType::RotationMatrix;
        };

        using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}

//namespace parameters_fast_learning{
//    namespace bpt = backprop_tools;
//    template<typename T, typename TI>
//    struct environment{
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
////        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
////        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_2<T>;
////        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_4<T>;
//        using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
//        using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;
//
//        using PARAMETERS_TYPE = backprop_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, 4, REWARD_FUNCTION>;
//        static constexpr PARAMETERS_TYPE parameters = {
//                backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie_old<T, TI, REWARD_FUNCTION>,
//                {0.01}, // integration dt
//                {
//                        backprop_tools::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
//                        reward_function,
//                        {   // Observation noise
//                            0, // position
//                            0, // orientation
//                            0, // linear_velocity
//                            0, // angular_velocity
//                        },
//                        {   // Action noise
//                            0, // std of additive gaussian noise onto the normalized action (-1, 1)
//                        },
////                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_simplified<T, TI, 4, REWARD_FUNCTION>,
////                        backprop_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
//                        backprop_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
//                },
//                typename PARAMETERS_TYPE::Disturbances{
//                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10}, // random_force;
//                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
//                }
//        };
//
//        using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;
//
//        struct ENVIRONMENT_STATIC_PARAMETERS: bpt::rl::environments::multirotor::StaticParametersDefault<TI>{
//            static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
//            static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
//            static constexpr bpt::rl::environments::multirotor::StateType STATE_TYPE = bpt::rl::environments::multirotor::StateType::Base;
//            static constexpr bpt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = bpt::rl::environments::multirotor::ObservationType::Normal;
//        };
//
//        using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
//        using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
//    };
//}
