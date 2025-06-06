#include <rl_tools/rl/environments/pendulum/pendulum.h>
#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/models.h>
#include <rl_tools/rl/algorithms/td3/td3.h>
#include <rl_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <rl_tools/utils/generic/typing.h>

template<typename DEVICE, typename T>
struct parameters_pendulum_0{
    using TI = typename DEVICE::index_t;
    struct env{
        using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
        using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
    };

    template <typename ENVIRONMENT>
    struct rl{
        struct ACTOR_CRITIC_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
            constexpr static TI CRITIC_BATCH_SIZE = 100;
            constexpr static TI ACTOR_BATCH_SIZE = 100;
            static constexpr TI N_WARMUP_STEPS_ACTOR = ACTOR_BATCH_SIZE;
            static constexpr TI N_WARMUP_STEPS_CRITIC = CRITIC_BATCH_SIZE;
        };
        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH>;
        using CRITIC_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY>;

        using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<T, TI>;
        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
        using ACTOR_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using ACTOR_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_CAPABILITY, ACTOR_SPEC>;
        using ACTOR_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<rlt::nn::capability::Forward, ACTOR_SPEC>;

        using CRITIC_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::Adam, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;
        using CRITIC_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_CAPABILITY, CRITIC_SPEC>;
        using CRITIC_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<rlt::nn::capability::Forward, CRITIC_SPEC>;


        using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, ACTOR_CRITIC_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        struct OFF_POLICY_RUNNER_PARAMETERS: rlt::rl::components::off_policy_runner::ParametersDefault<T, TI>{
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI REPLAY_BUFFER_CAPACITY = 500000;
            static constexpr TI EPISODE_STEP_LIMIT = 200;
        };
        using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, OFF_POLICY_RUNNER_PARAMETERS>;
        using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
        using CRITIC_BATCH_TYPE = rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
        using ACTOR_BATCH_TYPE = rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
        using CRITIC_TRAINING_BUFFERS_TYPE = rlt::rl::algorithms::td3::CriticTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;
        using ACTOR_TRAINING_BUFFERS_TYPE = rlt::rl::algorithms::td3::ActorTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;

    };
};

