#include <backprop_tools/rl/environments/pendulum/pendulum.h>
#include <backprop_tools/rl/environments/pendulum/operations_generic.h>
#include <backprop_tools/nn_models/models.h>
#include <backprop_tools/rl/algorithms/td3/td3.h>
#include <backprop_tools/rl/components/off_policy_runner/off_policy_runner.h>

#include <backprop_tools/utils/generic/typing.h>

template<typename DEVICE, typename T>
struct parameters_pendulum_0{
    using TI = typename DEVICE::index_t;
    struct env{
        using PENDULUM_SPEC = bpt::rl::environments::pendulum::Specification<T, TI, bpt::rl::environments::pendulum::DefaultParameters<T>>;
        using ENVIRONMENT = bpt::rl::environments::Pendulum<PENDULUM_SPEC>;
    };

    template <typename ENVIRONMENT>
    struct rl{
        struct ACTOR_CRITIC_PARAMETERS: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
            constexpr static TI CRITIC_BATCH_SIZE = 100;
            constexpr static TI ACTOR_BATCH_SIZE = 100;
            static constexpr TI N_WARMUP_STEPS_ACTOR = ACTOR_BATCH_SIZE;
            static constexpr TI N_WARMUP_STEPS_CRITIC = CRITIC_BATCH_SIZE;
        };
        using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::TANH, ACTOR_CRITIC_PARAMETERS::ACTOR_BATCH_SIZE>;
        using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, bpt::nn::activation_functions::RELU, bpt::nn::activation_functions::IDENTITY, ACTOR_CRITIC_PARAMETERS::CRITIC_BATCH_SIZE>;

        using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T, TI>;
        using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_NETWORK_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

        using ACTOR_TARGET_NETWORK_SPEC = bpt::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

        using CRITIC_NETWORK_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

        using CRITIC_TARGET_NETWORK_SPEC = backprop_tools::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TARGET_NETWORK_TYPE = backprop_tools::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;


        using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, ACTOR_CRITIC_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

        static constexpr TI N_ENVIRONMENTS = 1;
        static constexpr TI REPLAY_BUFFER_CAP = 500000;
        static constexpr TI ENVIRONMENT_STEP_LIMIT = 200;
        using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, bpt::rl::components::off_policy_runner::DefaultParameters<T> >;
        using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
        using CRITIC_BATCH_TYPE = bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
        using ACTOR_BATCH_TYPE = bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
        using CRITIC_TRAINING_BUFFERS_TYPE = bpt::rl::algorithms::td3::CriticTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;
        using ACTOR_TRAINING_BUFFERS_TYPE = bpt::rl::algorithms::td3::ActorTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;

    };
};

