#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/operations_generic.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/operations_generic.h>


#ifndef RL_TOOLS_BENCHMARK
#include <rl_tools/rl/utils/evaluation.h>
#include <chrono>
#endif

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEV_SPEC = rlt::devices::DefaultCPUSpecification;
using DEVICE = rlt::devices::CPU<DEV_SPEC>;

struct TrainingConfig{
    using T = float;
    using TI = typename DEVICE::index_t;
    using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));

    typedef rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
    typedef rlt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

    struct DEVICE_SPEC: rlt::devices::DefaultCPUSpecification {
        using LOGGING = rlt::devices::logging::CPU;
    };
    struct SACPendulumParameters: rlt::rl::algorithms::sac::DefaultParameters<T, TI>{
        constexpr static TI CRITIC_BATCH_SIZE = 100;
        constexpr static TI ACTOR_BATCH_SIZE = 100;
    };

    using SAC_PARAMETERS = SACPendulumParameters;

    using ActorStructureSpec = rlt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM, 2*ENVIRONMENT::ACTION_DIM, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::TANH, SAC_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CriticStructureSpec = rlt::nn_models::mlp::StructureSpecification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, rlt::nn::activation_functions::RELU, rlt::nn::activation_functions::IDENTITY, SAC_PARAMETERS::CRITIC_BATCH_SIZE>;


    using OPTIMIZER_SPEC = typename rlt::nn::optimizers::adam::Specification<T, TI>;
    using OPTIMIZER = rlt::nn::optimizers::Adam<OPTIMIZER_SPEC>;
    using ACTOR_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
    using ACTOR_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = rlt::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
    using CRITIC_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = rlt::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = rlt::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

    using ALPHA_PARAMETER_TYPE = rlt::nn::parameters::Adam;
    using ACTOR_CRITIC_SPEC = rlt::rl::algorithms::sac::Specification<T, TI, ENVIRONMENT, ACTOR_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, ALPHA_PARAMETER_TYPE, OPTIMIZER, OPTIMIZER, OPTIMIZER, SAC_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = rlt::rl::algorithms::sac::ActorCritic<ACTOR_CRITIC_SPEC>;


    static constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
#ifndef RL_TOOLS_STEP_LIMIT
    static constexpr TI STEP_LIMIT = 100000; //2 * N_WARMUP_STEPS;
#else
    static constexpr TI STEP_LIMIT = RL_TOOLS_STEP_LIMIT;
#endif
    static constexpr TI EVALUATION_INTERVAL = 1000;
    static constexpr TI REPLAY_BUFFER_CAP = STEP_LIMIT;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
    static constexpr bool COLLECT_EPISODE_STATS = true;
    static constexpr TI EPISODE_STATS_BUFFER_SIZE = 1000;
    static constexpr bool STOCHASTIC_POLICY = true;
    using OFF_POLICY_RUNNER_SPEC = rlt::rl::components::off_policy_runner::Specification<
            T,
            TI,
            ENVIRONMENT,
            1,
            false,
            REPLAY_BUFFER_CAP,
            EPISODE_STEP_LIMIT,
            STOCHASTIC_POLICY,
            COLLECT_EPISODE_STATS,
            EPISODE_STATS_BUFFER_SIZE
    >;
    const T STATE_TOLERANCE = 0.00001;
    static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
};

template <typename T_TRAINING_CONFIG>
struct CoreTrainingState{
    using TRAINING_CONFIG = T_TRAINING_CONFIG;
    using T = typename TRAINING_CONFIG::T;
    using TI = typename TRAINING_CONFIG::TI;
    typename TRAINING_CONFIG::OPTIMIZER actor_optimizer, critic_optimizers[2];
    typename TRAINING_CONFIG::RNG rng;
    rlt::rl::environments::DummyUI ui;
    rlt::rl::components::OffPolicyRunner<typename TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
    typename TRAINING_CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE actor_critic;
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffer<1> actor_deterministic_evaluation_buffers;
    rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    rlt::rl::algorithms::sac::CriticTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_SPEC> critic_training_buffers;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, TRAINING_CONFIG::SAC_PARAMETERS::CRITIC_BATCH_SIZE, TRAINING_CONFIG::ENVIRONMENT::ACTION_DIM>> action_noise_critic[2];
    rlt::MatrixDynamic<rlt::matrix::Specification<typename TRAINING_CONFIG::T, TI, 1, TRAINING_CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
    typename TRAINING_CONFIG::CRITIC_NETWORK_TYPE::template Buffer<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    rlt::rl::components::off_policy_runner::Batch<rlt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, TRAINING_CONFIG::SAC_PARAMETERS::CRITIC_BATCH_SIZE, TRAINING_CONFIG::ENVIRONMENT::ACTION_DIM>> action_noise_actor;
    rlt::rl::algorithms::sac::ActorTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffer<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffer<TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
};

template <typename TRAINING_CONFIG>
struct TrainingState: CoreTrainingState<TRAINING_CONFIG>{
    using T = typename TRAINING_CONFIG::T;
    using TI = typename TRAINING_CONFIG::TI;
#ifndef RL_TOOLS_BENCHMARK
    std::chrono::high_resolution_clock::time_point start_time;
#endif
    TI step = 0;
#ifndef RL_TOOLS_BENCHMARK
    static constexpr TI N_EVALUATIONS = TRAINING_CONFIG::STEP_LIMIT / TRAINING_CONFIG::EVALUATION_INTERVAL;
    static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
    T evaluation_returns[N_EVALUATIONS];
#endif
};


template <typename DEVICE, typename TRAINING_STATE>
void training_init(DEVICE& device, TRAINING_STATE& ts, typename TRAINING_STATE::TRAINING_CONFIG::TI seed){
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;


    ts.rng = rlt::random::default_engine(typename DEVICE::SPEC::RANDOM(), seed);

    rlt::malloc(device, ts.actor_critic);
    rlt::init(device, ts.actor_critic, ts.rng);

    rlt::malloc(device, ts.off_policy_runner);
    rlt::init(device, ts.off_policy_runner, ts.envs);

    rlt::malloc(device, ts.critic_batch);
    rlt::malloc(device, ts.critic_training_buffers);
    rlt::malloc(device, ts.action_noise_critic[0]);
    rlt::malloc(device, ts.action_noise_critic[1]);
    rlt::malloc(device, ts.critic_buffers[0]);
    rlt::malloc(device, ts.critic_buffers[1]);

    rlt::malloc(device, ts.actor_batch);
    rlt::malloc(device, ts.actor_training_buffers);
    rlt::malloc(device, ts.action_noise_actor);
    rlt::malloc(device, ts.actor_buffers_eval);
    rlt::malloc(device, ts.actor_buffers[0]);
    rlt::malloc(device, ts.actor_buffers[1]);

    rlt::malloc(device, ts.observations_mean);
    rlt::malloc(device, ts.observations_std);

    rlt::malloc(device, ts.actor_deterministic_evaluation_buffers);

    rlt::set_all(device, ts.observations_mean, 0);
    rlt::set_all(device, ts.observations_std, 1);


#ifndef RL_TOOLS_BENCHMARK
    ts.start_time = std::chrono::high_resolution_clock::now();
#endif
    ts.step = 0;
}

template <typename DEVICE, typename TRAINING_STATE>
void training_destroy(DEVICE& device, TRAINING_STATE& ts){
    rlt::free(device, ts.critic_batch);
    rlt::free(device, ts.critic_training_buffers);
    rlt::free(device, ts.actor_batch);
    rlt::free(device, ts.actor_training_buffers);
    rlt::free(device, ts.off_policy_runner);
    rlt::free(device, ts.actor_critic);
    rlt::free(device, ts.observations_mean);
    rlt::free(device, ts.observations_std);
}


template <typename DEVICE, typename TRAINING_STATE>
bool training_step(DEVICE& device, TRAINING_STATE& ts){
    bool finished = false;
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;
    rlt::step(device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
#ifndef RL_TOOLS_BENCHMARK
    if(ts.step % 1000 == 0){
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - ts.start_time;
        std::cout << "step_i: " << ts.step << " " << elapsed_seconds.count() << "s" << std::endl;
    }
#endif
    if(ts.step > TRAINING_CONFIG::N_WARMUP_STEPS){

        for(int critic_i = 0; critic_i < 2; critic_i++){
            rlt::gather_batch(device, ts.off_policy_runner, ts.critic_batch, ts.rng);
            rlt::randn(device, ts.action_noise_critic[critic_i], ts.rng);
            rlt::train_critic(device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers, ts.action_noise_critic[critic_i]);
        }

        {
            rlt::gather_batch(device, ts.off_policy_runner, ts.actor_batch, ts.rng);
            rlt::randn(device, ts.action_noise_actor, ts.rng);
            rlt::train_actor(device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.action_noise_actor);
        }

        rlt::update_critic_targets(device, ts.actor_critic);
    }
#ifndef RL_TOOLS_BENCHMARK
    if(ts.step % TRAINING_CONFIG::EVALUATION_INTERVAL == 0){
        auto result = rlt::evaluate(device, ts.envs[0], ts.ui, ts.actor_critic.actor, rlt::rl::utils::evaluation::Specification<1, TRAINING_CONFIG::EPISODE_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng);
        std::cout << "Mean return: " << result.returns_mean << std::endl;
        ts.evaluation_returns[ts.step / TRAINING_CONFIG::EVALUATION_INTERVAL] = result.returns_mean;
    }
#endif
    if(ts.step > 5000){
        ts.off_policy_runner.parameters.exploration_noise = 0;
    }
    ts.step++;
    if(ts.step > TRAINING_CONFIG::STEP_LIMIT){
#ifndef RL_TOOLS_BENCHMARK
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - ts.start_time;
        std::cout << "total time: " << elapsed_seconds.count() << "s" << std::endl;
#endif
        return true;
    }
    else{
        return finished;
    }
}
