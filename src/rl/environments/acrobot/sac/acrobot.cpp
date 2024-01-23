#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <rl_tools/rl/environments/acrobot/operations_generic.h>
#ifdef RL_TOOLS_ENABLE_GTK
#include <rl_tools/rl/environments/acrobot/ui.h>
#endif

#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

namespace training_config{
    using namespace rlt::nn_models::sequential::interface;
    struct TrainingConfig{
        using LOGGER = rlt::LOGGER_FACTORY<>;
        using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
//    using DEVICE = rlt::devices::CPU<DEV_SPEC>;
        using DEVICE = rlt::DEVICE_FACTORY<DEV_SPEC>;
        using T = float;
        using TI = typename DEVICE::index_t;

//        using ENV_PARAMETERS = rlt::rl::environments::acrobot::DefaultParameters<T>;
        using ENV_PARAMETERS = rlt::rl::environments::acrobot::EasyParameters<T>;
        using ENV_SPEC = rlt::rl::environments::acrobot::Specification<T, TI, ENV_PARAMETERS>;
        using ENVIRONMENT = rlt::rl::environments::AcrobotSwingup<ENV_SPEC>;
//        using UI_SPEC = rlt::rl::environments::acrobot::ui::Specification<T, TI, ENVIRONMENT, 500, 500>;
//        using UI = rlt::rl::environments::acrobot::UI<UI_SPEC>;

        using UI = bool;

        struct DEVICE_SPEC: rlt::devices::DefaultCPUSpecification {
            using LOGGING = rlt::devices::logging::CPU;
        };
        struct SACPendulumParameters: rlt::rl::algorithms::sac::DefaultParameters<T, DEVICE::index_t, ENVIRONMENT::ACTION_DIM>{
            constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
            constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
            static constexpr T TARGET_ENTROPY = -1;
        };

        using SAC_PARAMETERS = SACPendulumParameters;


        template <typename PARAMETER_TYPE, template<typename> class LAYER_TYPE = rlt::nn::layers::dense::LayerBackwardGradient>
        struct ACTOR{
            static constexpr TI HIDDEN_DIM = 64;
            static constexpr TI BATCH_SIZE = SAC_PARAMETERS::ACTOR_BATCH_SIZE;
            using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_1 = LAYER_TYPE<LAYER_1_SPEC>;
            using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_2 = LAYER_TYPE<LAYER_2_SPEC>;
            static constexpr TI ACTOR_OUTPUT_DIM = ENVIRONMENT::ACTION_DIM * 2; // to express mean and log_std for each action
            using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ACTOR_OUTPUT_DIM, rlt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>; // note the output activation should be identity because we want to sample from a gaussian and then squash afterwards (taking into account the squashing in the distribution)
            using LAYER_3 = LAYER_TYPE<LAYER_3_SPEC>;

            using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
        };

        template <typename PARAMETER_TYPE, template<typename> class LAYER_TYPE = rlt::nn::layers::dense::LayerBackwardGradient>
        struct CRITIC{
            static constexpr TI HIDDEN_DIM = 64;
            static constexpr TI BATCH_SIZE = SAC_PARAMETERS::CRITIC_BATCH_SIZE;

            using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_1 = LAYER_TYPE<LAYER_1_SPEC>;
            using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_2 = LAYER_TYPE<LAYER_2_SPEC>;
            using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
            using LAYER_3 = LAYER_TYPE<LAYER_3_SPEC>;

            using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
        };

        using OPTIMIZER_SPEC = rlt::nn::optimizers::adam::Specification<T, typename DEVICE::index_t>;

        using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
        using ALPHA_OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;

        using ACTOR_TYPE = ACTOR<rlt::nn::parameters::Adam>::MODEL;
        using ACTOR_TARGET_TYPE = ACTOR<rlt::nn::parameters::Adam, rlt::nn::layers::dense::Layer>::MODEL;
        using CRITIC_TYPE = CRITIC<rlt::nn::parameters::Adam>::MODEL;
        using CRITIC_TARGET_TYPE = CRITIC<rlt::nn::parameters::Adam, rlt::nn::layers::dense::Layer>::MODEL;

        using ALPHA_PARAMETER_TYPE = rlt::nn::parameters::Adam;

        using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::sac::Specification<T, DEVICE::index_t, ENVIRONMENT, ACTOR_TYPE, ACTOR_TARGET_TYPE, CRITIC_TYPE, CRITIC_TARGET_TYPE, ALPHA_PARAMETER_TYPE, OPTIMIZER, OPTIMIZER, ALPHA_OPTIMIZER, SAC_PARAMETERS>;
        using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::sac::ActorCritic<ACTOR_CRITIC_SPEC>;

        static constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
        static constexpr DEVICE::index_t STEP_LIMIT = 500000; //2 * N_WARMUP_STEPS;
        static constexpr bool DETERMINISTIC_EVALUATION = true;
        static constexpr DEVICE::index_t EVALUATION_INTERVAL = 1000;
        static constexpr TI NUM_EVALUATION_EPISODES = 10;
        static constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = STEP_LIMIT;
        static constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 100;
        static constexpr bool COLLECT_EPISODE_STATS = false;
        static constexpr DEVICE::index_t EPISODE_STATS_BUFFER_SIZE = 1000;
        using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
                T,
                DEVICE::index_t,
                ENVIRONMENT,
                1,
                false,
                REPLAY_BUFFER_CAP,
                ENVIRONMENT_STEP_LIMIT,
                rlt::rl::components::off_policy_runner::DefaultParameters<T>,
                true,
                COLLECT_EPISODE_STATS,
                EPISODE_STATS_BUFFER_SIZE
        >;
        static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
    };
}

int main(){
    using CONFIG = typename training_config::TrainingConfig;
    using T = typename CONFIG::T;
    using TI = typename CONFIG::TI;
    rlt::rl::algorithms::sac::loop::TrainingState<CONFIG> ts;
//    rlt::init(ts.device, ts.envs[0]);
    rlt::rl::algorithms::sac::loop::init(ts, 1);
//    ts.off_policy_runner.parameters.exploration_noise = 0.5;
//    ts.envs[0].parameters.dt = 0.01;
    for(TI step_i=0; step_i < CONFIG::STEP_LIMIT; step_i++){
        rlt::rl::algorithms::sac::loop::step(ts);
//        rlt::set_state(ts.device, ts.envs[0], ts.ui, get(ts.off_policy_runner.states, 0, 0));
//        rlt::render(ts.device, ts.envs[0], ts.ui);
        if(step_i % 1000 == 0){
            std::cout << "step: " << step_i << std::endl;
            rlt::MatrixStatic<rlt::matrix::Specification<T, TI, 1, CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observation_mean, observation_std;
            rlt::set_all(ts.device, observation_mean, 0);
            rlt::set_all(ts.device, observation_std, 0);
            bool no_ui = false;
            auto result = rlt::evaluate(ts.device, ts.envs[0], no_ui, ts.actor_critic.actor, rlt::rl::utils::evaluation::Specification<10, 100>{}, observation_mean, observation_std, ts.actor_buffers_eval, ts.rng_eval);
            std::cout << "Evaluation result: " << result.returns_mean << " +/- " << result.returns_std << std::endl;
        }
    }
    return 0;
}