
#include <layer_in_c/rl/environments/operations_generic.h>

#include <layer_in_c/nn_models/models.h>
#include <layer_in_c/rl/algorithms/td3/td3.h>
#include <layer_in_c/rl/components/off_policy_runner/off_policy_runner.h>

#include <layer_in_c/utils/generic/typing.h>

template<typename DEVICE, typename T>
struct parameters_0{

    using TI = typename DEVICE::index_t;
    using PENDULUM_SPEC = lic::rl::environments::pendulum::Specification<T, TI, lic::rl::environments::pendulum::DefaultParameters<T>>;
    using ENVIRONMENT = lic::rl::environments::Pendulum<PENDULUM_SPEC>;

    struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<T, TI>{
        constexpr static TI CRITIC_BATCH_SIZE = 100;
        constexpr static TI ACTOR_BATCH_SIZE = 100;
    };

    using TD3_PARAMETERS = TD3PendulumParameters;

    using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;

    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC, typename lic::nn::optimizers::adam::DefaultParametersTorch<T>>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ACTOR_STRUCTURE_SPEC>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC, typename lic::nn::optimizers::adam::DefaultParametersTorch<T>>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CRITIC_STRUCTURE_SPEC>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

    using ACTOR_CRITIC_SPEC = lic::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = lic::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;

    static constexpr TI N_ENVIRONMENTS = 32;
    static constexpr TI REPLAY_BUFFER_CAP = 500000;
    static constexpr TI ENVIRONMENT_STEP_LIMIT = 200;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, lic::rl::components::off_policy_runner::DefaultParameters<T> >;
    using OFF_POLICY_RUNNER_TYPE = lic::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
    using CRITIC_BATCH_TYPE = lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>>;
    using ACTOR_BATCH_TYPE = lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<OFF_POLICY_RUNNER_SPEC, ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>>;
    using CRITIC_TRAINING_BUFFERS_TYPE = lic::rl::algorithms::td3::CriticTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;
    using ACTOR_TRAINING_BUFFERS_TYPE = lic::rl::algorithms::td3::ActorTrainingBuffers<typename ACTOR_CRITIC_TYPE::SPEC>;

    static constexpr TI N_WARMUP_STEPS = ACTOR_CRITIC_SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
};
