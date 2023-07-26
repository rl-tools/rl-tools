#include <backprop_tools/rl/environments/pendulum/operations_cpu.h>
#include <backprop_tools/rl/algorithms/ppo/ppo.h>
#include <backprop_tools/rl/components/on_policy_runner/on_policy_runner.h>
namespace parameters_0{
    template <typename T, typename TI>
    struct environment{
        using ENVIRONMENT_SPEC = bpt::rl::environments::pendulum::Specification<T, TI>;
        using ENVIRONMENT = bpt::rl::environments::Pendulum<ENVIRONMENT_SPEC>;
    };
    template <typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        static constexpr TI BATCH_SIZE = 64;
        using ACTOR_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, bpt::nn::activation_functions::ActivationFunction::TANH, bpt::nn::activation_functions::IDENTITY, BATCH_SIZE>;

        struct OPTIMIZER_PARAMETERS: bpt::nn::optimizers::adam::DefaultParametersTorch<T, TI>{
            static constexpr T ALPHA = 0.001;
        };
        using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_SPEC = bpt::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TYPE = bpt::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
        using CRITIC_STRUCTURE_SPEC = bpt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 64, bpt::nn::activation_functions::ActivationFunction::TANH, bpt::nn::activation_functions::IDENTITY, BATCH_SIZE>;
        using CRITIC_SPEC = bpt::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TYPE = bpt::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

        struct PPO_PARAMETERS: bpt::rl::algorithms::ppo::DefaultParameters<T, TI>{
            static constexpr TI N_EPOCHS = 1;
        };
        using PPO_SPEC = bpt::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
        using PPO_TYPE = bpt::rl::algorithms::PPO<PPO_SPEC>;
        using PPO_BUFFERS_TYPE = bpt::rl::algorithms::ppo::Buffers<PPO_SPEC>;

        static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 200;
        static constexpr TI N_ENVIRONMENTS = 10;
        using ON_POLICY_RUNNER_SPEC = bpt::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
        using ON_POLICY_RUNNER_TYPE = bpt::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 200;
        using ON_POLICY_RUNNER_DATASET_SPEC = bpt::rl::components::on_policy_runner::DatasetSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
        using ON_POLICY_RUNNER_DATASET_TYPE = bpt::rl::components::on_policy_runner::Dataset<ON_POLICY_RUNNER_DATASET_SPEC>;

        using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
        using ACTOR_BUFFERS = typename ACTOR_TYPE::template Buffers<BATCH_SIZE>;
        using CRITIC_BUFFERS = typename CRITIC_TYPE::template Buffers<BATCH_SIZE>;
        using CRITIC_BUFFERS_ALL = typename CRITIC_TYPE::template Buffers<ON_POLICY_RUNNER_DATASET_SPEC::STEPS_TOTAL_ALL>;
    };
}
