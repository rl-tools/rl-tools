#include "environment.h"
#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>


RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::zoo::flag::ppo{
    template<typename TYPE_POLICY, typename TI, typename ENVIRONMENT, typename PARAMETERS, bool DYNAMIC_ALLOCATION=true>
    struct ConfigApproximatorsGRU{
        static constexpr bool USE_GRU = true;
        using PPO_PARAMETERS = typename PARAMETERS::PPO_PARAMETERS;
        template <typename CAPABILITY>
        struct Actor{
            using INPUT_SHAPE = tensor::Shape<TI, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, PARAMETERS::N_ENVIRONMENTS, ENVIRONMENT::Observation::DIM>;
            using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
            using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
            using INPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>;
            using INPUT_LAYER = nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
            using GRU_SPEC = nn::layers::gru::Configuration<TYPE_POLICY, TI, PARAMETERS::ACTOR_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU = nn::layers::gru::BindConfiguration<GRU_SPEC>;
            using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM, PARAMETERS::ACTOR_NUM_LAYERS, PARAMETERS::ACTOR_HIDDEN_DIM, PARAMETERS::ACTOR_ACTIVATION_FUNCTION,  nn::activation_functions::IDENTITY>;
            using MLP = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;

            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
            using MODULE = Module<STANDARDIZATION_LAYER, Module<INPUT_LAYER, Module<GRU, Module<MLP>>>>;
            using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>;
        };
        template <typename CAPABILITY>
        struct Critic{
            using INPUT_SHAPE = tensor::Shape<TI, PARAMETERS::ON_POLICY_RUNNER_STEPS_PER_ENV, PARAMETERS::N_ENVIRONMENTS, ENVIRONMENT::ObservationPrivileged::DIM>;
            using STANDARDIZATION_LAYER_CONFIG = nn::layers::standardize::Configuration<TYPE_POLICY, TI>;
            using STANDARDIZATION_LAYER = nn::layers::standardize::BindConfiguration<STANDARDIZATION_LAYER_CONFIG>;
            using INPUT_LAYER_CONFIG = nn::layers::dense::Configuration<TYPE_POLICY, TI, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::layers::dense::DefaultInitializer<TYPE_POLICY, TI>, nn::parameters::groups::Input>;
            using INPUT_LAYER = nn::layers::dense::BindConfiguration<INPUT_LAYER_CONFIG>;
            using GRU_SPEC = nn::layers::gru::Configuration<TYPE_POLICY, TI, PARAMETERS::CRITIC_HIDDEN_DIM, nn::parameters::groups::Normal, true>;
            using GRU = nn::layers::gru::BindConfiguration<GRU_SPEC>;
            using CONFIG = nn_models::mlp::Configuration<TYPE_POLICY, TI, 1, PARAMETERS::CRITIC_NUM_LAYERS, PARAMETERS::CRITIC_HIDDEN_DIM, PARAMETERS::CRITIC_ACTIVATION_FUNCTION, nn::activation_functions::IDENTITY>;
            using TYPE = nn_models::mlp_unconditional_stddev::BindConfiguration<CONFIG>;
            template <typename T_CONTENT, typename T_NEXT_MODULE = nn_models::sequential::OutputModule>
            using Module = typename nn_models::sequential::Module<T_CONTENT, T_NEXT_MODULE>;
            using MODULE = Module<STANDARDIZATION_LAYER, Module<INPUT_LAYER, Module<GRU, Module<TYPE>>>>;
            using MODEL = nn_models::sequential::Build<CAPABILITY, MODULE, INPUT_SHAPE>;
        };

        using ACTOR_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
        using CRITIC_OPTIMIZER_SPEC = nn::optimizers::adam::Specification<TYPE_POLICY, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS, DYNAMIC_ALLOCATION>;
        using ACTOR_OPTIMIZER = nn::optimizers::Adam<ACTOR_OPTIMIZER_SPEC>;
        using CRITIC_OPTIMIZER = nn::optimizers::Adam<CRITIC_OPTIMIZER_SPEC>;
        using CAPABILITY_ADAM = nn::capability::Gradient<nn::parameters::Adam, DYNAMIC_ALLOCATION>;
        using ACTOR_TYPE = typename Actor<CAPABILITY_ADAM>::MODEL;
        using CRITIC_TYPE = typename Critic<CAPABILITY_ADAM>::MODEL;
    };
    namespace rlt = rl_tools;
    template <typename DEVICE, typename TYPE_POLICY, typename TI, typename RNG, bool DYNAMIC_ALLOCATION>
    struct FACTORY{
        using T = typename TYPE_POLICY::DEFAULT;
        using ENVIRONMENT = typename ENVIRONMENT_FACTORY<DEVICE, TYPE_POLICY, TI>::ENVIRONMENT;
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
            static constexpr TI N_ENVIRONMENTS = 64;
            static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 256;
            static constexpr TI BATCH_SIZE = N_ENVIRONMENTS*ON_POLICY_RUNNER_STEPS_PER_ENV;
            static constexpr TI TOTAL_STEP_LIMIT = 100000000;
            static constexpr TI ACTOR_HIDDEN_DIM = 128;
            static constexpr TI CRITIC_HIDDEN_DIM = 128;
            static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
            static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
            static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
            struct OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
                static constexpr T ALPHA = 0.0003;
            };
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
            struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
                static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
                static constexpr TI N_EPOCHS = 1;
                static constexpr T GAMMA = 0.995;
                static constexpr T LAMBDA = 0.95;
                static constexpr T INITIAL_ACTION_STD = 1.0;
                static constexpr bool SHUFFLE_EPOCH = false;
                static constexpr bool STATEFUL_ACTOR_AND_CRITIC = true;
            };
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsGRU, DYNAMIC_ALLOCATION>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
