

#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/abs_exp.h>
#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/squared.h>
#include <layer_in_c/rl/environments/multirotor/parameters/reward_functions/default.h>
#include <layer_in_c/rl/environments/multirotor/parameters/dynamics/crazy_flie.h>
#include <layer_in_c/rl/environments/multirotor/parameters/init/default.h>
#include <layer_in_c/rl/environments/multirotor/parameters/termination/default.h>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

namespace parameters_0{

    template<typename DEVICE, typename T>
    struct environment{
        using TI = typename DEVICE::index_t;
        static constexpr auto reward_function = layer_in_c::rl::environments::multirotor::parameters::reward_functions::reward_dr<T>;
        using REWARD_FUNCTION = decltype(reward_function);

        static constexpr layer_in_c::rl::environments::multirotor::Parameters<T, TI, 4, REWARD_FUNCTION> parameters = {
                layer_in_c::rl::environments::multirotor::parameters::dynamics::crazy_flie<T, TI, REWARD_FUNCTION>,
                {0.01}, // integration dt
                {
                        layer_in_c::rl::environments::multirotor::parameters::init::all_around<T, TI, 4, REWARD_FUNCTION>,
                        layer_in_c::rl::environments::multirotor::parameters::termination::classic<T, TI, 4, REWARD_FUNCTION>,
                        reward_function,
                }
        };

        using ENVIRONMENT_SPEC = lic::rl::environments::multirotor::Specification<T, typename DEVICE::index_t, decltype(parameters), lic::rl::environments::multirotor::StaticParameters>;
        using ENVIRONMENT = lic::rl::environments::Multirotor<ENVIRONMENT_SPEC>;
    };


    template<typename DEVICE, typename T, typename ENVIRONMENT>
    struct rl{
        using TI = typename DEVICE::index_t;
        struct TD3_PARAMETERS: lic::rl::algorithms::td3::DefaultParameters<T, typename DEVICE::index_t>{
            static constexpr typename DEVICE::index_t ACTOR_BATCH_SIZE = 256;
            static constexpr typename DEVICE::index_t CRITIC_BATCH_SIZE = 256;
            static constexpr typename DEVICE::index_t CRITIC_TRAINING_INTERVAL = 10;
            static constexpr typename DEVICE::index_t ACTOR_TRAINING_INTERVAL = 20;
            static constexpr typename DEVICE::index_t CRITIC_TARGET_UPDATE_INTERVAL = 10;
            static constexpr typename DEVICE::index_t ACTOR_TARGET_UPDATE_INTERVAL = 20;
            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.25;
            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
            static constexpr bool IGNORE_TERMINATION = false;

        };

        struct ReplayBufferParameters: lic::rl::components::off_policy_runner::DefaultParameters<T>{
            static constexpr T EXPLORATION_NOISE = 0.2;
        };

        using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 256, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;

        using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<T>>;
        using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

        using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
        using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

        using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec, typename lic::nn::optimizers::adam::DefaultParametersTorch<T>>;
        using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

        using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
        using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

        using TD3_SPEC = lic::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
        using ActorCriticType = lic::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

        static constexpr TI REPLAY_BUFFER_CAP = 5000000;
        static constexpr TI ENVIRONMENT_STEP_LIMIT = 1000;
        using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<
                T,
                TI,
                ENVIRONMENT,
                REPLAY_BUFFER_CAP,
                ENVIRONMENT_STEP_LIMIT,
                ReplayBufferParameters
        >;
        static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
    };


}
