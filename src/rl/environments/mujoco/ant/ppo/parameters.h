#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <rl_tools/rl/algorithms/ppo/ppo.h>
#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
#include <rl_tools/nn/layers/standardize/layer.h>
#include <rl_tools/nn_models/sequential/model.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/network.h>
namespace parameters_0{
    template <typename T, typename TI>
    struct environment{
        using ENVIRONMENT_PARAMETERS = rlt::rl::environments::mujoco::ant::DefaultParameters<T, TI>;
        using ENVIRONMENT_SPEC = rlt::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    };
    using namespace rlt;
    template <typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        static constexpr TI BATCH_SIZE = 2048;
//        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY, BATCH_SIZE>;

        template <typename CAPABILITY>
        struct Actor{
            using ACTOR_SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, ENVIRONMENT::ACTION_DIM, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, nn::activation_functions::IDENTITY>;
            using ACTOR_TYPE = nn_models::mlp_unconditional_stddev::BindSpecification<ACTOR_SPEC>;
            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using ACTOR_MODULE = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork>;
            using STANDARDIZATION_LAYER_SPEC = nn::layers::standardize::Specification<T, TI, ENVIRONMENT::Observation::DIM>;
            using STANDARDIZATION_LAYER = nn::layers::standardize::BindSpecification<STANDARDIZATION_LAYER_SPEC>;
            using MODEL = typename IF::template Module<STANDARDIZATION_LAYER::template Layer, ACTOR_MODULE>;
//            using MODEL = ACTOR_MODULE;
        };
        template <typename CAPABILITY>
        struct Critic{
            using SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, 1, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, nn::activation_functions::IDENTITY>;
            using TYPE = nn_models::mlp_unconditional_stddev::BindSpecification<SPEC>;
            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using ACTOR_MODULE = typename IF::template Module<TYPE::template NeuralNetwork>;
            using STANDARDIZATION_LAYER_SPEC = nn::layers::standardize::Specification<T, TI, ENVIRONMENT::Observation::DIM>;
            using STANDARDIZATION_LAYER = nn::layers::standardize::BindSpecification<STANDARDIZATION_LAYER_SPEC>;
            using MODEL = typename IF::template Module<STANDARDIZATION_LAYER::template Layer, ACTOR_MODULE>;
//            using MODEL = ACTOR_MODULE;
        };

        using ACTOR_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
        using CRITIC_OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, TI>;
        using ACTOR_OPTIMIZER = rlt::nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
        using CRITIC_OPTIMIZER = rlt::nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
        using CAPABILITY_ADAM = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
//        using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<CAPABILITY_ADAM, ACTOR_SPEC>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
//        using ACTOR_TYPE_INFERENCE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<rlt::nn::layer_capability::Forward, ACTOR_SPEC>;
//        using CRITIC_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::Observation::DIM, 1, 3, 256, rlt::nn::activation_functions::ActivationFunction::RELU, rlt::nn::activation_functions::IDENTITY, BATCH_SIZE>;
//        using CRITIC_TYPE = rlt::nn_models::mlp::NeuralNetwork<CAPABILITY_ADAM, CRITIC_SPEC>;
        using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;

        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
            static constexpr TI N_EPOCHS = 4;
            static constexpr bool LEARN_ACTION_STD = true;
            static constexpr T INITIAL_ACTION_STD = 0.5;
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
            static constexpr bool NORMALIZE_ADVANTAGE = false;
            static constexpr T GAMMA = 0.99;
            static constexpr bool ADAPTIVE_LEARNING_RATE = true;
            static constexpr T ADAPTIVE_LEARNING_RATE_POLICY_KL_THRESHOLD = 0.008;

            static constexpr bool NORMALIZE_OBSERVATIONS = true;
        };
        static constexpr T OBSERVATION_NORMALIZATION_WARMUP_STEPS = PPO_PARAMETERS::NORMALIZE_OBSERVATIONS ? 1 : 0;
        using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
        using PPO_TYPE = rlt::rl::algorithms::PPO<PPO_SPEC>;
        using PPO_BUFFERS_TYPE = rlt::rl::algorithms::ppo::Buffers<PPO_SPEC>;

        static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 1000;
        static constexpr TI N_ENVIRONMENTS = 64;
        using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
        using ON_POLICY_RUNNER_TYPE = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
        using ON_POLICY_RUNNER_DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
        using ON_POLICY_RUNNER_DATASET_TYPE = rlt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;


        using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffer<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
        using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffer<BATCH_SIZE>;
        using CRITIC_BUFFERS = typename CRITIC_TYPE::template Buffer<BATCH_SIZE>;
        using CRITIC_BUFFERS_GAE = typename CRITIC_TYPE::template Buffer<ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
    };
}
