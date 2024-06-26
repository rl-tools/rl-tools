#include <rl_tools/rl/environments/mujoco/ant/ant.h>

#include <rl_tools/nn_models/models.h>
#include <rl_tools/rl/algorithms/td3/td3.h>
#include <rl_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <rl_tools/utils/generic/typing.h>

namespace parameters_0{

    template<typename T, typename TI>
    struct environment{
        using ENVIRONMENT_SPEC = rlt::rl::environments::mujoco::ant::Specification<T, TI, rlt::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
        using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    };

    template<typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        struct ACTOR_CRITIC_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
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

        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH>;
        using CRITIC_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 256, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;

        using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;

        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
        using ACTOR_CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using ACTOR_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_CAPABILITY, ACTOR_SPEC>;
        using ACTOR_TARGET_TYPE = rlt::nn_models::mlp::NeuralNetwork<rlt::nn::layer_capability::Forward, ACTOR_SPEC>;

        using CRITIC_CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;
        using CRITIC_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_CAPABILITY , CRITIC_SPEC>;
        using CRITIC_TARGET_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_CAPABILITY, CRITIC_SPEC>;

        using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, ACTOR_CRITIC_PARAMETERS>;
        using ActorCriticType = rlt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

//        static constexpr TI N_ENVIRONMENTS = 1;
//        static constexpr TI REPLAY_BUFFER_CAP = 1000000;
        static constexpr TI EPISODE_STEP_LIMIT = 1000;
        struct OFF_POLICY_RUNNER_PARAMETERS: rlt::rl::components::off_policy_runner::ParametersDefault<T, TI>{
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI REPLAY_BUFFER_CAPACITY = 1000000;
            static constexpr TI EPISODE_STEP_LIMIT = 1000;
            static constexpr bool COLLECT_EPISODE_STATS = true;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
            static constexpr T EXPLORATION_NOISE = 0.1;
        };
        using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, OFF_POLICY_RUNNER_PARAMETERS>;

        static constexpr TI N_WARMUP_STEPS_CRITIC = 10000;
        static constexpr TI N_WARMUP_STEPS_ACTOR = 10000;
    };


}
