#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>
#include <layer_in_c/rl/algorithms/ppo/ppo.h>
#include <layer_in_c/rl/components/on_policy_runner/on_policy_runner.h>
namespace parameters_0{
    template <typename T, typename TI>
    struct environment{
        using ENVIRONMENT_PARAMETERS = lic::rl::environments::mujoco::ant::DefaultParameters<T, TI>;
        using ENVIRONMENT_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
        using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    };
    template <typename T, typename TI, typename ENVIRONMENT>
    struct rl{
        static constexpr TI BATCH_SIZE = 2048;
        using ACTOR_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 256, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;

        struct OPTIMIZER_PARAMETERS: lic::nn::optimizers::adam::DefaultParametersTorch<T>{
            static constexpr T ALPHA = 3e-4;
        };
        using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
        using ACTOR_SPEC = lic::nn_models::mlp::AdamSpecification<ACTOR_STRUCTURE_SPEC>;
        using ACTOR_TYPE = lic::nn_models::mlp_unconditional_stddev::NeuralNetworkAdam<ACTOR_SPEC>;
        using ACTOR_TYPE_INFERENCE = lic::nn_models::mlp_unconditional_stddev::NeuralNetwork<ACTOR_SPEC>;
        using CRITIC_STRUCTURE_SPEC = lic::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 1, 3, 256, lic::nn::activation_functions::ActivationFunction::RELU, lic::nn::activation_functions::IDENTITY, BATCH_SIZE>;
        using CRITIC_SPEC = lic::nn_models::mlp::AdamSpecification<CRITIC_STRUCTURE_SPEC>;
        using CRITIC_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<CRITIC_SPEC>;

        struct PPO_PARAMETERS: lic::rl::algorithms::ppo::DefaultParameters<T, TI>{
            static constexpr TI N_EPOCHS = 4;
            static constexpr bool LEARN_ACTION_STD = true;
            static constexpr T INITIAL_ACTION_STD = 0.5;
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0;
            static constexpr bool NORMALIZE_ADVANTAGE = false;
        };
        using PPO_SPEC = lic::rl::algorithms::ppo::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, CRITIC_TYPE, PPO_PARAMETERS>;
        using PPO_TYPE = lic::rl::algorithms::PPO<PPO_SPEC>;
        using PPO_BUFFERS_TYPE = lic::rl::algorithms::ppo::Buffers<PPO_SPEC>;

        static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 1000;
        static constexpr TI N_ENVIRONMENTS = 64;
        using ON_POLICY_RUNNER_SPEC = lic::rl::components::on_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ON_POLICY_RUNNER_STEP_LIMIT>;
        using ON_POLICY_RUNNER_TYPE = lic::rl::components::OnPolicyRunner<ON_POLICY_RUNNER_SPEC>;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 64;
        using ON_POLICY_RUNNER_BUFFER_SPEC = lic::rl::components::on_policy_runner::BufferSpecification<ON_POLICY_RUNNER_SPEC, ON_POLICY_RUNNER_STEPS_PER_ENV>;
        using ON_POLICY_RUNNER_BUFFER_TYPE = lic::rl::components::on_policy_runner::Buffer<ON_POLICY_RUNNER_BUFFER_SPEC>;


        using ACTOR_EVAL_BUFFERS = typename ACTOR_TYPE::template Buffers<ON_POLICY_RUNNER_SPEC::N_ENVIRONMENTS>;
        using ACTOR_BUFFERS = typename ACTOR_TYPE::template BuffersForwardBackward<BATCH_SIZE>;
        using CRITIC_BUFFERS = typename CRITIC_TYPE::template BuffersForwardBackward<BATCH_SIZE>;
        using CRITIC_BUFFERS_GAE = typename CRITIC_TYPE::template BuffersForwardBackward<ON_POLICY_RUNNER_BUFFER_SPEC::STEPS_TOTAL_ALL>;
    };
}
