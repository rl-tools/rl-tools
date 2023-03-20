#include <layer_in_c/rl/environments/mujoco/ant/ant.h>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/utils/generic/typing.h>

namespace parameters_0{

    template<typename T, typename TI>
    struct environment{
        using ENVIRONMENT_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, lic::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
        using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    };

    template<typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        struct ACTOR_CRITIC_PARAMETERS: lic::rl::algorithms::td3::DefaultParameters<T, TI>{
            static constexpr TI ACTOR_BATCH_SIZE = 256;
            static constexpr TI CRITIC_BATCH_SIZE = 256;
            static constexpr TI CRITIC_TRAINING_INTERVAL = 1;
            static constexpr TI ACTOR_TRAINING_INTERVAL = 2;
            static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 2;
            static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 2;
            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
            static constexpr bool IGNORE_TERMINATION = false;
        };

        struct OFF_POLICY_RUNNER_PARAMETERS: lic::rl::components::off_policy_runner::DefaultParameters<T>{
            static constexpr T EXPLORATION_NOISE = 0.1;
        };

        using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 256, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 256, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;

        struct OPTIMIZER_PARAMETERS: lic::nn::optimizers::adam::DefaultParametersTorch<T>{
            static constexpr T ALPHA = 1e-3;
        };

        using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

        using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

        using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

        using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

        using ACTOR_CRITIC_SPEC = lic::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, ACTOR_CRITIC_PARAMETERS>;
        using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI REPLAY_BUFFER_CAP = 1000000;
        static constexpr TI ENVIRONMENT_STEP_LIMIT = 1000;
        using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, OFF_POLICY_RUNNER_PARAMETERS, true, 1000>;

        static constexpr TI N_WARMUP_STEPS_CRITIC = 10000;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 10000;
    };


}
