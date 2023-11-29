#ifdef RL_TOOLS_TEST_BARE
#include <rl_tools/operations/arm.h>
#include <rl_tools/nn/operations_generic.h>
#else
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#endif
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;




#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/algorithms/td3/operations_generic.h>


#include <rl_tools/rl/utils/evaluation.h>

#ifndef RL_TOOLS_TEST_BARE
#include <chrono>
#endif


#ifdef RL_TOOLS_TEST_BARE
using DEVICE = rlt::devices::DefaultARM; // The arm device just uses plain C++
#else
using LOGGER = rlt::LOGGER_FACTORY<>;
using DEV_SPEC = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, LOGGER>;
using DEVICE = rlt::DEVICE_FACTORY<DEV_SPEC>;
#endif



using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;


struct TD3_PENDULUM_PARAMETERS: rlt::rl::algorithms::td3::DefaultParameters<T, TI>{
    constexpr static TI CRITIC_BATCH_SIZE = 100;
    constexpr static TI ACTOR_BATCH_SIZE = 100;
};

using TD3_PARAMETERS = TD3_PENDULUM_PARAMETERS;

namespace function_approximators{ // to simplify the model definition we import the sequential interface but we don't want to pollute the global namespace hence we do it in a model definition namespace
    using namespace rlt::nn_models::sequential::interface;

    template <typename PARAMETER_TYPE>
    struct ACTOR{
        static constexpr TI HIDDEN_DIM = 64;
        static constexpr TI BATCH_SIZE = TD3_PARAMETERS::ACTOR_BATCH_SIZE;
        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ENVIRONMENT::ACTION_DIM, rlt::nn::activation_functions::ActivationFunction::TANH, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

        using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
    };

    template <typename PARAMETER_TYPE>
    struct CRITIC{
        static constexpr TI HIDDEN_DIM = 64;
        static constexpr TI BATCH_SIZE = TD3_PARAMETERS::CRITIC_BATCH_SIZE;

        using LAYER_1_SPEC = rlt::nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_1 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_1_SPEC>;
        using LAYER_2_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, rlt::nn::activation_functions::ActivationFunction::RELU, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_2 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_2_SPEC>;
        using LAYER_3_SPEC = rlt::nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, rlt::nn::activation_functions::ActivationFunction::IDENTITY, PARAMETER_TYPE, BATCH_SIZE>;
        using LAYER_3 = rlt::nn::layers::dense::LayerBackwardGradient<LAYER_3_SPEC>;

        using MODEL = Module<LAYER_1, Module<LAYER_2, Module<LAYER_3>>>;
    };
}

using OPTIMIZER_PARAMETERS = typename rlt::nn::optimizers::adam::DefaultParametersTorch<T, TI>;
using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
using ACTOR_NETWORK_TYPE = typename function_approximators::ACTOR<rlt::nn::parameters::Adam>::MODEL;

using ACTOR_TARGET_NETWORK_TYPE = typename function_approximators::ACTOR<rlt::nn::parameters::Plain>::MODEL;

using CRITIC_NETWORK_TYPE = typename function_approximators::CRITIC<rlt::nn::parameters::Adam>::MODEL;

using CRITIC_TARGET_NETWORK_TYPE = typename function_approximators::CRITIC<rlt::nn::parameters::Plain>::MODEL;

using TD3_SPEC = rlt::rl::algorithms::td3::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, OPTIMIZER, TD3_PARAMETERS>;
using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::td3::ActorCritic<TD3_SPEC>;

#ifdef RL_TOOLS_TEST_RL_ALGORITHMS_TD3_FULL_TRAINING_DEBUG
constexpr TI STEP_LIMIT = 1000;
#else
constexpr TI STEP_LIMIT = 10000;
#endif

constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
constexpr TI EPISODE_STEP_LIMIT = 200;
using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
        T,
        TI,
        ENVIRONMENT,
        1,
        false,
        REPLAY_BUFFER_CAP,
        EPISODE_STEP_LIMIT,
        rlt::rl::components::off_policy_runner::DefaultParameters<T>
>;
using OFF_POLICY_RUNNER_TYPE = rlt::rl::components::OffPolicyRunner<OFF_POLICY_RUNNER_SPEC>;
constexpr T STATE_TOLERANCE = 0.00001;
constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);

TI run(TI seed=0){
    TI return_code = 100;
    DEVICE device;

    auto rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);

    OFF_POLICY_RUNNER_TYPE off_policy_runner;
    ACTOR_CRITIC_TYPE actor_critic;

    rlt::malloc(device, actor_critic);
    rlt::init(device, actor_critic, rng);

    bool ui = false;

    rlt::construct(device, device.logger);

    rlt::malloc(device, off_policy_runner);
    ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    rlt::init(device, off_policy_runner, envs);

    OFF_POLICY_RUNNER_TYPE::Batch<TD3_PARAMETERS::CRITIC_BATCH_SIZE> critic_batch;
    rlt::rl::algorithms::td3::CriticTrainingBuffers<ACTOR_CRITIC_TYPE::SPEC> critic_training_buffers;
    CRITIC_NETWORK_TYPE::DoubleBuffer<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    rlt::malloc(device, critic_batch);
    rlt::malloc(device, critic_training_buffers);
    rlt::malloc(device, critic_buffers[0]);
    rlt::malloc(device, critic_buffers[1]);

    OFF_POLICY_RUNNER_TYPE::Batch<TD3_PARAMETERS::ACTOR_BATCH_SIZE> actor_batch;
    rlt::rl::algorithms::td3::ActorTrainingBuffers<ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    ACTOR_NETWORK_TYPE::DoubleBuffer<ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    ACTOR_NETWORK_TYPE::DoubleBuffer<OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    rlt::malloc(device, actor_batch);
    rlt::malloc(device, actor_training_buffers);
    rlt::malloc(device, actor_buffers_eval);
    rlt::malloc(device, actor_buffers[0]);
    rlt::malloc(device, actor_buffers[1]);

    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_mean;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::OBSERVATION_DIM>> observations_std;
    rlt::malloc(device, observations_mean);
    rlt::malloc(device, observations_std);
    rlt::set_all(device, observations_mean, 0);
    rlt::set_all(device, observations_std, 1);


#ifndef RL_TOOLS_TEST_BARE
    auto start_time = std::chrono::high_resolution_clock::now();
#endif

    for(int step_i = 0; step_i < STEP_LIMIT; step_i+=OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS){
        rlt::set_step(device, device.logger, step_i);
        rlt::step(device, off_policy_runner, actor_critic.actor, actor_buffers_eval, rng);

        if(step_i > N_WARMUP_STEPS){
            if(step_i % 1000 == 0){
#ifndef RL_TOOLS_TEST_BARE
                auto current_time = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_seconds = current_time - start_time;
                std::cout << "step_i: " << step_i << " " << elapsed_seconds.count() << "s" << std::endl;
#endif
            }

            for(int critic_i = 0; critic_i < 2; critic_i++){
                rlt::target_action_noise(device, actor_critic, critic_training_buffers.target_next_action_noise, rng);
                rlt::gather_batch(device, off_policy_runner, critic_batch, rng);
                rlt::train_critic(device, actor_critic, critic_i == 0 ? actor_critic.critic_1 : actor_critic.critic_2, critic_batch, actor_critic.critic_optimizers[critic_i], actor_buffers[critic_i], critic_buffers[critic_i], critic_training_buffers);
            }

            if(step_i % 2 == 0){
                {
                    rlt::gather_batch(device, off_policy_runner, actor_batch, rng);
                    rlt::train_actor(device, actor_critic, actor_batch, actor_critic.actor_optimizer, actor_buffers[0], critic_buffers[0], actor_training_buffers);
                }

                rlt::update_critic_targets(device, actor_critic);
                rlt::update_actor_target(device, actor_critic);
            }
        }
#ifndef RL_TOOLS_RL_ENVIRONMENTS_PENDULUM_DISABLE_EVALUATION
        if(step_i % 1000 == 0){
            auto result = rlt::evaluate(device, envs[0], ui, actor_critic.actor, rlt::rl::utils::evaluation::Specification<10, EPISODE_STEP_LIMIT>{}, observations_mean, observations_std, actor_buffers_eval, rng);
#ifndef RL_TOOLS_TEST_BARE
            std::cout << "Mean return: " << result.returns_mean << std::endl;
#endif
            rlt::add_scalar(device, device.logger, "mean_return", result.returns_mean);
            return_code = rlt::math::min(DEVICE::SPEC::MATH(), (TI)(-result.returns_mean / 100), return_code);
        }
#else
        return_code = 0;
#endif
    }
    {
#ifndef RL_TOOLS_TEST_BARE
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
        // expect 2.25s for 10k steps with all optimizations @ Lenovo P1 Intel(R) Core(TM) i9-10885H
#endif
    }
    rlt::free(device, critic_batch);
    rlt::free(device, critic_training_buffers);
    rlt::free(device, actor_batch);
    rlt::free(device, actor_training_buffers);
    rlt::free(device, off_policy_runner);
    rlt::free(device, actor_critic);
    rlt::free(device, observations_mean);
    rlt::free(device, observations_std);

    rlt::destruct(device, device.logger);
    return return_code;
}

