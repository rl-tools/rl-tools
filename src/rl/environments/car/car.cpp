//#include <rl_tools/operations/cpu_mux.h>
//#include <rl_tools/nn/operations_cpu_mux.h>
//
//#include <rl_tools/rl/environments/car/operations_cpu.h>
//#if RL_TOOLS_ENABLE_GTK
//#include <rl_tools/rl/environments/car/ui.h>
//#endif
//
//#include <rl_tools/nn_models/operations_generic.h>
//
//#include <rl_tools/rl/algorithms/td3/loop.h>
//namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
//
//namespace training_config {
//    using namespace rlt::nn_models::sequential::interface; // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
//    struct Config{
//#if !defined(RL_TOOLS_ENABLE_TENSORBOARD) || defined(RL_TOOLS_DISABLE_TENSORBOARD)
//        using DEV_SPEC = rlt::devices::DefaultCPUSpecification;
//#else
//        using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, rlt::devices::logging::CPU_TENSORBOARD<>>;
//#endif
////    using DEVICE = rlt::devices::CPU<DEV_SPEC>;
//        using DEVICE = rlt::devices::DEVICE_FACTORY<DEV_SPEC>;
//        using T = float;
//        using TI = typename DEVICE::index_t;
//
//        using ENV_SPEC = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 20>;
//        using ENVIRONMENT = rlt::rl::environments::CarTrack<ENV_SPEC>;
//        using ENVIRONMENT_EVALUATION = ENVIRONMENT;
//#if RL_TOOLS_ENABLE_GTK
//        using UI = rlt::rl::environments::car::UI<rlt::rl::environments::car::ui::Specification<T, TI, ENVIRONMENT, 1000, 60>>;
//#else
//        using UI = bool;
//#endif
//
//        struct DEVICE_SPEC: rlt::devices::DefaultCPUSpecification {
//            using LOGGING = rlt::devices::logging::CPU;
//        };
//#if defined(RL_TOOLS_ENABLE_TENSORBOARD) && !defined(RL_TOOLS_DISABLE_TENSORBOARD)
//        static constexpr bool CONSTRUCT_LOGGER = true;
//#else
//        static constexpr bool CONSTRUCT_LOGGER = false;
//#endif
//        static constexpr T EXPLORATION_NOISE_MULTIPLE = 0.5;
//        struct TD3PendulumParameters: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
//            constexpr static TI CRITIC_BATCH_SIZE = 256;
//            constexpr static TI ACTOR_BATCH_SIZE = 256;
//            constexpr static T GAMMA = 0.997;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2 * EXPLORATION_NOISE_MULTIPLE;
//            static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5 * EXPLORATION_NOISE_MULTIPLE;
//        };
//
//        using TD3_PARAMETERS = TD3PendulumParameters;
//
//        template <typename PARAMETER_TYPE>
//        struct ACTOR{
//            static constexpr TI HIDDEN_DIM = 64;
//            static constexpr TI BATCH_SIZE = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
//            using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
//            using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
//            using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::TANH, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
//
//            using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
//        };
//
//        template <typename PARAMETER_TYPE>
//        struct CRITIC{
//            static constexpr TI HIDDEN_DIM = 64;
//            static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;
//
//            using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
//            using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
//            using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
//            using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;
//
//            using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
//        };
//
//
//
//        //using ACTOR_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
//        //using CRITIC_STRUCTURE_SPEC = rlt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;
//
//        using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<T, TI>;
//        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
//        using ACTOR_TYPE = typename ACTOR<rlt::nn::parameters::Adam>::MODEL;
//        using ACTOR_TARGET_TYPE = typename ACTOR<rlt::nn::parameters::Plain>::MODEL;
//        using CRITIC_TYPE = typename CRITIC<rlt::nn::parameters::Adam>::MODEL;
//        using CRITIC_TARGET_TYPE = typename CRITIC<rlt::nn::parameters::Plain>::MODEL;
//
//        using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, OPTIMIZER, TD3_PARAMETERS>;
//        using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;
//
//
//        static constexpr int N_WARMUP_STEPS_ACTOR = 10000; //ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
//        static constexpr int N_WARMUP_STEPS_CRITIC = 10000; //ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
//#ifndef RL_TOOLS_STEP_LIMIT
//        static constexpr TI STEP_LIMIT = 500000000; //2 * N_WARMUP_STEPS;
//#else
//        static constexpr TI STEP_LIMIT = RL_TOOLS_STEP_LIMIT;
//#endif
//        static constexpr bool DETERMINISTIC_EVALUATION = true;
//        static constexpr TI NUM_EVALUATION_EPISODES = 1;
//        static constexpr TI EVALUATION_INTERVAL = 20000;
//        static constexpr TI REPLAY_BUFFER_CAP = 1000000;
//        static constexpr TI ENVIRONMENT_STEP_LIMIT = 500;
//        static constexpr TI ENVIRONMENT_STEP_LIMIT_EVALUATION = 1000;
//        static constexpr bool COLLECT_EPISODE_STATS = true;
//        static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
//        using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
//                T,
//                TI,
//                ENVIRONMENT,
//                1,
//                false,
//                REPLAY_BUFFER_CAP,
//                ENVIRONMENT_STEP_LIMIT,
//                rlt::rl::components::off_policy_runner::DefaultParameters<T>,
//                false,
//                COLLECT_EPISODE_STATS,
//                EPISODE_STATS_BUFFER_SIZE
//        >;
//        const T STATE_TOLERANCE = 0.00001;
//        static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
//    };
//}
//
//int main(){
//    using CONFIG = typename training_config::Config;
//    using TI = typename CONFIG::TI ;
//    rlt::rl::algorithms::td3::loop::TrainingState<CONFIG> ts;
//    for(auto& env : ts.envs){
//        rlt::init(ts.device, env);
//    }
//    rlt::init(ts.device, ts.env_eval);
//    ts.off_policy_runner.parameters.exploration_noise *= CONFIG::EXPLORATION_NOISE_MULTIPLE;
//    rlt::rl::algorithms::td3::loop::init(ts, 5);
//    for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
//        rlt::rl::algorithms::td3::loop::step(ts);
//    }
//    return 0;
//}

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <rl_tools/rl/environments/car/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>


#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#include <rl_tools/rl/loop/steps/timing/operations.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

using ENV_SPEC = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 20>;
using ENVIRONMENT = rlt::rl::environments::CarTrack<ENV_SPEC>;
using ENVIRONMENT_EVALUATION = ENVIRONMENT;

static constexpr T EXPLORATION_NOISE_MULTIPLE = 0.5;
struct TD3_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
    constexpr static TI CRITIC_BATCH_SIZE = 256;
    constexpr static TI ACTOR_BATCH_SIZE = 256;
    constexpr static T GAMMA = 0.997;
    static constexpr T TARGET_NEXT_ACTION_NOISE_STD = 0.2 * EXPLORATION_NOISE_MULTIPLE;
    static constexpr T TARGET_NEXT_ACTION_NOISE_CLIP = 0.5 * EXPLORATION_NOISE_MULTIPLE;
};
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    using TD3_PARAMETERS = TD3_PARAMETERS;
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::td3::loop::core::ConfigApproximatorsMLP>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main(int argc, char** argv) {
    DEVICE device;
    TI seed = 0;
    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    while(!rlt::step(device, ts)){
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
    }
    rlt::free(device, ts);
    return 0;
}
