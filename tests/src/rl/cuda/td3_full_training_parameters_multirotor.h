#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <backprop_tools/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <backprop_tools/rl/environments/multirotor/parameters/init/default.h>
#include <backprop_tools/rl/environments/multirotor/parameters/termination/default.h>
#include <backprop_tools/rl/environments/multirotor/operations_generic.h>

#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/rl/algorithms/td3/td3.h>
#include <backprop_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <backprop_tools/utils/generic/typing.h>


template<typename DEVICE, typename T>
struct parameters_multirotor_0{
    struct env{
        using TI = typename DEVICE::index_t;
        static constexpr auto reward_function = backprop_tools::rl::environments::multirotor::parameters::reward_functions::reward_dr<T>;
        using REWARD_FUNCTION_CONST = typename backprop_tools::utils::typing::remove_cv_t<decltype(reward_function)>;
        using REWARD_FUNCTION = typename backprop_tools::utils::typing::remove_cv<REWARD_FUNCTION_CONST>::type;

        static constexpr backprop_tools::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION> parameters = {
                backprop_tools::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
                        backprop_tools::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
                        backprop_tools::rl::environments::multirotor::parameters::termination::classic<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                }
        };

        using PARAMETERS = typename backprop_tools::utils::typing::remove_cv_t<decltype(parameters)>;

        using ENVIRONMENT_SPEC = bpt::rl::environments::multirotor::Specification<T, typename DEVICE::index_t, PARAMETERS, bpt::rl::environments::multirotor::StaticParameters>;
        using ENVIRONMENT = bpt::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };

    template<typename ENVIRONMENT>
    struct rl{
        using TI = typename DEVICE::index_t;
        struct ACTOR_CRITIC_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<T, typename DEVICE::index_t>{
            static constexpr TI ACTOR_BATCH_SIZE = 256;
            static constexpr TI CRITIC_BATCH_SIZE = 256;
            static constexpr TI CRITIC_TRAINING_INTERVAL = 10;
            static constexpr TI ACTOR_TRAINING_INTERVAL = 20;
            static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 10;
            static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 20;
            static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.25;
            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
            static constexpr bool IGNORE_TERMINATION = false;
        };

        struct OFF_POLICY_RUNNER_PARAMETERS: bpt::rl::components::off_policy_runner::DefaultParameters<T>{
            static constexpr T EXPLORATION_NOISE = 0.2;
        };

        using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 256, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;

        using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T>;
        using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

        using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

        using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

        using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

        using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, ACTOR_CRITIC_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI REPLAY_BUFFER_CAP = 300000;
        static constexpr TI ENVIRONMENT_STEP_LIMIT = 1000;
        using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, OFF_POLICY_RUNNER_PARAMETERS>;
        using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
        using CRITIC_BATCH_TYPE = bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
        using ACTOR_BATCH_TYPE = bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
        using CRITIC_TRAINING_BUFFERS_TYPE = bpt::rl::algorithms::td3::CriticTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;
        using ACTOR_TRAINING_BUFFERS_TYPE = bpt::rl::algorithms::td3::ActorTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;

    };

};
