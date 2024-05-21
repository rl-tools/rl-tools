#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/rl/algorithms/ppo/ppo.h>
#include <rl_tools/rl/components/on_policy_runner/on_policy_runner.h>
namespace parameters_0{
    using namespace rlt;
    template <typename T, typename TI>
    struct environment{
        using ENVIRONMENT_SPEC = rlt::rl::environments::pendulum::Specification<T, TI>;
        using ENVIRONMENT = rlt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
    };
    template <typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        static constexpr TI BATCH_SIZE = 64;

        using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::Specification<T, TI>;
        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY, BATCH_SIZE>;
        using CAPABILITY_ADAM = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;

        template <typename CAPABILITY>
        struct Actor{
            using ACTOR_SPEC = nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, nn::activation_functions::IDENTITY, BATCH_SIZE>;
            using ACTOR_TYPE = nn_models::mlp_unconditional_stddev::BindSpecification<ACTOR_SPEC>;
            using IF = nn_models::sequential::Interface<CAPABILITY>;
            using MODEL = typename IF::template Module<ACTOR_TYPE::template NeuralNetwork>;
        };
//        using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
//        using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;
//        using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
//        using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;

        using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
//        using ACTOR_TYPE_INFERENCE = typename Actor<nn::layer_capability::Forward>::MODEL;
//        using ACTOR_TYPE = rlt::nn_models::mlp_unconditional_stddev::NeuralNetwork<CAPABILITY_ADAM, ACTOR_SPEC>;
        using CRITIC_SPEC = rlt::nn_models::mlp::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 64, rlt::nn::activation_functions::ActivationFunction::TANH, rlt::nn::activation_functions::IDENTITY, BATCH_SIZE>;
        using CRITIC_TYPE = rlt::nn_models::mlp::NeuralNetwork<CAPABILITY_ADAM, CRITIC_SPEC>;

        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
            static constexpr TI N_EPOCHS = 1;
        };
        using PPO_SPEC = rlt::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
        using PPO_TYPE = rlt::rl::algorithms::PPO<PPO_SPEC>;
        using PPO_BUFFERS_TYPE = rlt::rl::algorithms::ppo::Buffers<PPO_SPEC>;

        static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 200;
        static constexpr TI N_ENVIRONMENTS = 10;
        using ON_POLICY_RUNNER_SPEC = rlt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
        using ON_POLICY_RUNNER_TYPE = rlt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 200;
        using ON_POLICY_RUNNER_DATASET_SPEC = rlt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
        using ON_POLICY_RUNNER_DATASET_TYPE = rlt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;

        using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffer<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
        using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffer<BATCH_SIZE>;
        using CRITIC_BUFFERS = typename CRITIC_TYPE::template Buffer<BATCH_SIZE>;
        using CRITIC_BUFFERS_ALL = typename CRITIC_TYPE::template Buffer<ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
    };
}
