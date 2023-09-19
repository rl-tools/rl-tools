#include <backprop_tools/operations/cpu_mux.h>
#include <backprop_tools/nn/operations_cpu_mux.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>

#include <backprop_tools/nn_models/sequential/operations_generic.h>

#include <backprop_tools/rl/algorithms/td3/loop.h>
namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#include "../td3/parameters.h"
namespace multirotor_training{
    namespace config {
        using namespace bpt::nn_models::sequential::interface; // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
        struct Config{
            using DEV_SPEC = bpt::devices::DefaultCPUSpecification;
//    using DEVICE = bpt::devices::CPU<DEV_SPEC>;
            using DEVICE = bpt::DEVICE_FACTORY<DEV_SPEC>;
            using T = float;
            using TI = typename DEVICE::index_t;

            using ENVIRONMENT = parameters_0::environment<T, TI>::ENVIRONMENT;
            using UI = bool;

            struct DEVICE_SPEC: bpt::devices::DefaultCPUSpecification {
                using LOGGING = bpt::devices::logging::CPU;
            };
            struct TD3PendulumParameters: bpt::rl::algorithms::td3::DefaultParameters<T, TI>{
//                constexpr static typename TI CRITIC_BATCH_SIZE = 100;
//                constexpr static typename TI ACTOR_BATCH_SIZE = 100;
//                constexpr static T GAMMA = 0.997;
                static constexpr TI ACTOR_BATCH_SIZE = 256;
                static constexpr TI CRITIC_BATCH_SIZE = 256;
                static constexpr TI CRITIC_TRAINING_INTERVAL = 10;
                static constexpr TI ACTOR_TRAINING_INTERVAL = 20;
                static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = 10;
                static constexpr TI ACTOR_TARGET_UPDATE_INTERVAL = 20;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 1.0;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5;
                static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2;
                static constexpr T GAMMA = 0.99;
                static constexpr bool IGNORE_TERMINATION = false;
            };

            using TD3_PARAMETERS = TD3PendulumParameters;

            static constexpr bool ASYMMETRIC_OBSERVATIONS = ENVIRONMENT::PRIVILEGED_OBSERVATION_AVAILABLE;
            static constexpr TI CRITIC_OBSERVATION_DIM = ASYMMETRIC_OBSERVATIONS ? ENVIRONMENT::OBSERVATION_DIM_PRIVILEGED : ENVIRONMENT::OBSERVATION_DIM;

            template <typename PARAMETER_TYPE>
            struct ACTOR{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
                static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            template <typename PARAMETER_TYPE>
            struct CRITIC{
                static constexpr TI HIDDEN_DIM = 64;
                static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;

                static constexpr auto ACTIVATION_FUNCTION = bpt::nn::activation_functions::FAST_TANH;
                using LAYER_1_SPEC = bpt::nn::layers::dense::Specification<T, TI, CRITIC_OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_1 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
                using LAYER_2_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_2 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
                using LAYER_3_SPEC = bpt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, bpt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
                using LAYER_3 = bpt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

                using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
            };

            using OPTIMIZER_PARAMETERS = typename bpt::nn::optimizers::adam::DefaultParametersTorch<T, TI>;
            using OPTIMIZER = bpt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
            using ACTOR_TYPE = typename ACTOR<bpt::nn::parameters::Adam>::MODEL;
            using ACTOR_TARGET_TYPE = typename ACTOR<bpt::nn::parameters::Plain>::MODEL;
            using CRITIC_TYPE = typename CRITIC<bpt::nn::parameters::Adam>::MODEL;
            using CRITIC_TARGET_TYPE = typename CRITIC<bpt::nn::parameters::Plain>::MODEL;

            using ACTOR_CRITIC_SPEC = bpt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
            using ACTOR_CRITIC_TYPE = bpt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


            static constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
            static constexpr bool DETERMINISTIC_EVALUATION = true;
            static constexpr TI EVALUATION_INTERVAL = 50000;
            static constexpr TI NUM_EVALUATION_EPISODES = 10;
            static constexpr bool COLLECT_EPISODE_STATS = false;
            static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
            static constexpr TI N_ENVIRONMENTS = 1;
            static constexpr TI STEP_LIMIT = 1500001;
            static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
            static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
            using OFF_POLICY_RUNNER_SPEC = bpt::rl::components::off_policy_runner::Specification<T, TI, ENVIRONMENT, N_ENVIRONMENTS, ASYMMETRIC_OBSERVATIONS, REPLAY_BUFFER_CAP, ENVIRONMENT_STEP_LIMIT, bpt::rl::components::off_policy_runner::DefaultParameters<T>, false, true, 1000>;
            using OFF_POLICY_RUNNER_TYPE = bpt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
            static constexpr bpt::rl::components::off_policy_runner::DefaultParameters<T> off_policy_runner_parameters = {
                    0.5
            };

            static constexpr TI N_WARMUP_STEPS_CRITIC = 15000;
            static constexpr TI N_WARMUP_STEPS_ACTOR = 30000;
            static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
        };
    }

    namespace operations{
        using TrainingState = bpt::rl::algorithms::td3::loop::TrainingState<config::Config>;
        void init(TrainingState& ts){
            using TI = typename config::Config::TI;
            for (auto& env : ts.envs) {
                env.parameters = parameters_0::environment<config::Config::T, config::Config::TI>::parameters;
            }
            ts.eval_env.parameters = ts.envs[0].parameters;
            bpt::rl::algorithms::td3::loop::init(ts, 3);
            ts.off_policy_runner.parameters = parameters_0::rl<config::Config::T, config::Config::TI, config::Config::ENVIRONMENT>::off_policy_runner_parameters;
        }

        void step(TrainingState& ts){
            bpt::rl::algorithms::td3::loop::step(ts);
        }
    }
}
