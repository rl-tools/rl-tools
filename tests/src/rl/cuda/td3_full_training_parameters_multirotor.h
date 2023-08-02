#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <backprop_tools/rl/environments/multirotor/parameters/init/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/termination/default.h>

namespace parameters_sim2real{
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
                                0.001, // position
                                0.001, // orientation
                                0.002, // linear_velocity
                                0.002, // angular_velocity
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

namespace parameters_fast_learning{
    namespace bpt = backprop_tools;
    template<typename T, typename TI>
    struct environment{
        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_old_but_gold<T>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_mm<T, TI>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_2<T>;
//        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_squared_4<T>;
        using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        using PARAMETERS_TYPE = backprop_tools::rl::environments::multirotor::ParametersDisturbances<T, TI, 4, REWARD_FUNCTION>;
        static constexpr PARAMETERS_TYPE parameters = {
                backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie_old<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
                        backprop_tools::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                        {   // Observation noise
                                0, // position
                                0, // orientation
                                0, // linear_velocity
                                0, // angular_velocity
                        },
                        {   // Action noise
                                0, // std of additive gaussian noise onto the normalized action (-1, 1)
                        },
//                        backprop_tools::rl::environments::multirotor::parameters::init::all_around_simplified<T, TI, 4, REWARD_FUNCTION>,
//                        backprop_tools::rl::environments::multirotor::parameters::init::simple<T, TI, 4, REWARD_FUNCTION>,
                        backprop_tools::rl::environments::multirotor::parameters::termination::fast_learning<T, TI, 4, REWARD_FUNCTION>
                },
                typename PARAMETERS_TYPE::Disturbances{
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0.027 * 9.81 / 10}, // random_force;
                        typename PARAMETERS_TYPE::Disturbances::UnivariateGaussian{0, 0} // random_torque;
                }
        };

        using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;

        struct ENVIRONMENT_STATIC_PARAMETERS: bpt::rl::environments::multirotor::StaticParametersDefault<TI>{
            static constexpr bool ENFORCE_POSITIVE_QUATERNION = false;
            static constexpr bool RANDOMIZE_QUATERNION_SIGN = false;
            static constexpr bpt::rl::environments::multirotor::StateType STATE_TYPE = bpt::rl::environments::multirotor::StateType::Base;
            static constexpr bpt::rl::environments::multirotor::ObservationType OBSERVATION_TYPE = bpt::rl::environments::multirotor::ObservationType::Normal;
        };

        using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, TI, PARAMETERS, ENVIRONMENT_STATIC_PARAMETERS>;
        using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };
}

#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/rl/algorithms/td3/td3.h>
#include <backprop_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <backprop_tools/utils/generic/typing.h>

namespace parameters{
    template<typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        struct ACTOR_CRITIC_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
            static constexpr TI ACTOR_BATCH_SIZE = 256;
            static constexpr TI CRITIC_BATCH_SIZE = 256;
            static constexpr TI CRITIC_TRAINING_INTERVAL = 10;
            static constexpr TI ACTOR_TRAINING_INTERVAL = 20;
            static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 10;
            static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 20;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 1.0;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.5;
            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
            static constexpr T GAMMA = 0.99;
            static constexpr bool IGNORE_TERMINATION = false;
        };


        static constexpr bool ASYMMETRIC_OBSERVATIONS = true;
        static constexpr TI CRITIC_OBSERVATION_DIM = ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : ENVIRONMENT::OBSERVATION_DIM;
        static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
        using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, ACTIVATION_FUNCTION, bpt::nn::activation_functions::TANH, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, CRITIC_OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, ACTIVATION_FUNCTION, bpt::nn::activation_functions::IDENTITY, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;

        using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T, TI>;
        using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_SPEC>;

        using ACTOR_TARGET_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TARGET_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_SPEC>;

        using CRITIC_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

        using CRITIC_TARGET_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TARGET_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_SPEC>;

        using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, ACTOR_CRITIC_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI REPLAY_BUFFER_CAP = 500001;
        static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
        using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ASYMMETRIC_OBSERVATIONS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, bpt::rl::components::off_policy_runner::DefaultParameters<T>, true, 1000>;
        using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
        static constexpr bpt::rl::components::off_policy_runner::DefaultParameters<T> off_policy_runner_parameters = {
                0.5
        };

        static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
    };
}

namespace parameters_fast_learning{
    using parameters::rl;
    using parameters_fast_learning::environment;
}
namespace parameters_sim2real{
    using parameters::rl;
    using parameters_sim2real::environment;
}


namespace parameters_0 = parameters_sim2real;
//namespace parameters_0 = parameters_fast_learning;
