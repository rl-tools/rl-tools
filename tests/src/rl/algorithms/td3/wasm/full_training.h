#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/nn/operations_generic.h>

#include <layer_in_c/rl/environments/operations_generic.h>
#include <layer_in_c/nn_models/operations_generic.h>
#include <layer_in_c/rl/operations_generic.h>


#ifndef LAYER_IN_C_BENCHMARK
#include <layer_in_c/rl/utils/evaluation.h>
#include <chrono>
#endif



struct TrainingConfig{
    using DEV_SPEC = lic::devices::DefaultCPUSpecification;
    using DEVICE = lic::devices::CPU<DEV_SPEC>;
    using DTYPE = float;

    typedef lic::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, lic::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
    typedef lic::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;

    struct DEVICE_SPEC: lic::devices::DefaultCPUSpecification {
        using LOGGING = lic::devices::logging::CPU;
    };
    struct TD3PendulumParameters: lic::rl::algorithms::td3::DefaultParameters<DTYPE, DEVICE::index_t>{
        constexpr static typename DEVICE::index_t CRITIC_BATCH_SIZE = 100;
        constexpr static typename DEVICE::index_t ACTOR_BATCH_SIZE = 100;
    };

    using TD3_PARAMETERS = TD3PendulumParameters;

    using ActorStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM, ENVIRONMENT::ACTION_DIM, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::TANH, TD3_PARAMETERS::ACTOR_BATCH_SIZE>;
    using CriticStructureSpec = lic::nn_models::mlp::StructureSpecification<DTYPE, DEVICE::index_t, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, 1, 3, 64, lic::nn::activation_functions::RELU, lic::nn::activation_functions::IDENTITY, TD3_PARAMETERS::CRITIC_BATCH_SIZE>;


    using OPTIMIZER_PARAMETERS = typename lic::nn::optimizers::adam::DefaultParametersTorch<DTYPE>;
    using OPTIMIZER = lic::nn::optimizers::Adam<OPTIMIZER_PARAMETERS>;
    using ACTOR_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<ActorStructureSpec>;
    using ACTOR_NETWORK_TYPE = lic::nn_models::mlp::NeuralNetworkAdam<ACTOR_NETWORK_SPEC>;

    using ACTOR_TARGET_NETWORK_SPEC = lic::nn_models::mlp::InferenceSpecification<ActorStructureSpec>;
    using ACTOR_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<ACTOR_TARGET_NETWORK_SPEC>;

    using CRITIC_NETWORK_SPEC = lic::nn_models::mlp::AdamSpecification<CriticStructureSpec>;
    using CRITIC_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetworkAdam<CRITIC_NETWORK_SPEC>;

    using CRITIC_TARGET_NETWORK_SPEC = layer_in_c::nn_models::mlp::InferenceSpecification<CriticStructureSpec>;
    using CRITIC_TARGET_NETWORK_TYPE = layer_in_c::nn_models::mlp::NeuralNetwork<CRITIC_TARGET_NETWORK_SPEC>;

    using ACTOR_CRITIC_SPEC = lic::rl::algorithms::td3::Specification<DTYPE, DEVICE::index_t, ENVIRONMENT, ACTOR_NETWORK_TYPE, ACTOR_TARGET_NETWORK_TYPE, CRITIC_NETWORK_TYPE, CRITIC_TARGET_NETWORK_TYPE, TD3_PARAMETERS>;
    using ACTOR_CRITIC_TYPE = lic::rl::algorithms::td3::ActorCritic<ACTOR_CRITIC_SPEC>;


    static constexpr int N_WARMUP_STEPS = ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE;
#ifndef LAYER_IN_C_STEP_LIMIT
    static constexpr DEVICE::index_t STEP_LIMIT = 100000; //2 * N_WARMUP_STEPS;
#else
    static constexpr DEVICE::index_t STEP_LIMIT = LAYER_IN_C_STEP_LIMIT;
#endif
    static constexpr DEVICE::index_t EVALUATION_INTERVAL = 1000;
    static constexpr typename DEVICE::index_t REPLAY_BUFFER_CAP = STEP_LIMIT;
    static constexpr typename DEVICE::index_t ENVIRONMENT_STEP_LIMIT = 200;
    static constexpr bool COLLECT_EPISODE_STATS = true;
    static constexpr DEVICE::index_t EPISODE_STATS_BUFFER_SIZE = 1000;
    using OFF_POLICY_RUNNER_SPEC = lic::rl::components::off_policy_runner::Specification<
            DTYPE,
            DEVICE::index_t,
            ENVIRONMENT,
            1,
            REPLAY_BUFFER_CAP,
            ENVIRONMENT_STEP_LIMIT,
            lic::rl::components::off_policy_runner::DefaultParameters<DTYPE>,
            COLLECT_EPISODE_STATS,
            EPISODE_STATS_BUFFER_SIZE
    >;
    const DTYPE STATE_TOLERANCE = 0.00001;
    static_assert(ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE == ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE);
};

template <typename T_TRAINING_CONFIG>
struct CoreTrainingState{
    using TRAINING_CONFIG = T_TRAINING_CONFIG;
    using DEVICE = typename TRAINING_CONFIG::DEVICE;
    using TI = typename DEVICE::index_t;
    typename DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    typename TRAINING_CONFIG::OPTIMIZER optimizer;
    decltype(lic::random::default_engine(typename DEVICE::SPEC::RANDOM())) rng;
    bool ui = false;
    lic::rl::components::OffPolicyRunner<typename TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
    typename TRAINING_CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE actor_critic;
    lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    lic::rl::algorithms::td3::CriticTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_SPEC> critic_training_buffers;
    lic::MatrixDynamic<lic::matrix::Specification<typename TRAINING_CONFIG::DTYPE, TI, 1, TRAINING_CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
    typename TRAINING_CONFIG::CRITIC_NETWORK_TYPE::template BuffersForwardBackward<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    lic::rl::components::off_policy_runner::Batch<lic::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
    lic::rl::algorithms::td3::ActorTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffers<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffers<TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
};

template <typename TRAINING_CONFIG>
struct TrainingState: CoreTrainingState<TRAINING_CONFIG>{
    using T = typename TRAINING_CONFIG::DTYPE;
    using TI = typename TRAINING_CONFIG::DEVICE::index_t;
#ifndef LAYER_IN_C_BENCHMARK
    std::chrono::high_resolution_clock::time_point start_time;
#endif
    TI step = 0;
#ifndef LAYER_IN_C_BENCHMARK
    static constexpr TI N_EVALUATIONS = TRAINING_CONFIG::STEP_LIMIT / TRAINING_CONFIG::EVALUATION_INTERVAL;
    static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
    T evaluation_returns[N_EVALUATIONS];
#endif
};


template <typename TRAINING_STATE>
void training_init(TRAINING_STATE& ts){
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;

    ts.device.logger = &ts.logger;

    lic::malloc(ts.device, ts.actor_critic);
    lic::init(ts.device, ts.actor_critic, ts.optimizer, ts.rng);

    lic::malloc(ts.device, ts.off_policy_runner);
    lic::init(ts.device, ts.off_policy_runner, ts.envs);

    lic::malloc(ts.device, ts.critic_batch);
    lic::malloc(ts.device, ts.critic_training_buffers);
    lic::malloc(ts.device, ts.critic_buffers[0]);
    lic::malloc(ts.device, ts.critic_buffers[1]);

    lic::malloc(ts.device, ts.actor_batch);
    lic::malloc(ts.device, ts.actor_training_buffers);
    lic::malloc(ts.device, ts.actor_buffers_eval);
    lic::malloc(ts.device, ts.actor_buffers[0]);
    lic::malloc(ts.device, ts.actor_buffers[1]);

    lic::malloc(ts.device, ts.observations_mean);
    lic::malloc(ts.device, ts.observations_std);

    lic::set_all(ts.device, ts.observations_mean, 0);
    lic::set_all(ts.device, ts.observations_std, 1);


#ifndef LAYER_IN_C_BENCHMARK
    ts.start_time = std::chrono::high_resolution_clock::now();
#endif
    ts.step = 0;
}

template <typename TRAINING_STATE>
void training_destroy(TRAINING_STATE& ts){
    lic::free(ts.device, ts.critic_batch);
    lic::free(ts.device, ts.critic_training_buffers);
    lic::free(ts.device, ts.actor_batch);
    lic::free(ts.device, ts.actor_training_buffers);
    lic::free(ts.device, ts.off_policy_runner);
    lic::free(ts.device, ts.actor_critic);
    lic::free(ts.device, ts.observations_mean);
    lic::free(ts.device, ts.observations_std);
}


template <typename TRAINING_STATE>
bool training_step(TRAINING_STATE& ts){
    bool finished = false;
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;
    lic::step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
#ifndef LAYER_IN_C_BENCHMARK
    if(ts.step % 1000 == 0){
        auto current_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_seconds = current_time - ts.start_time;
        std::cout << "step_i: " << ts.step << " " << elapsed_seconds.count() << "s" << std::endl;
    }
#endif
    if(ts.step > TRAINING_CONFIG::N_WARMUP_STEPS){

        for(int critic_i = 0; critic_i < 2; critic_i++){
            lic::target_action_noise(ts.device, ts.actor_critic, ts.critic_training_buffers.target_next_action_noise, ts.rng);
            lic::gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
            lic::train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.optimizer, ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
        }

        if(ts.step % 2 == 0){
            {
                lic::gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                lic::train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers);
            }

            lic::update_critic_targets(ts.device, ts.actor_critic);
            lic::update_actor_target(ts.device, ts.actor_critic);
        }
    }
#ifndef LAYER_IN_C_BENCHMARK
    if(ts.step % TRAINING_CONFIG::EVALUATION_INTERVAL == 0){
        auto result = lic::evaluate(ts.device, ts.envs[0], ts.ui, ts.actor_critic.actor, lic::rl::utils::evaluation::Specification<1, TRAINING_CONFIG::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.rng, true);
        std::cout << "Mean return: " << result.mean << std::endl;
        ts.evaluation_returns[ts.step / TRAINING_CONFIG::EVALUATION_INTERVAL] = result.mean;
    }
#endif
    ts.step++;
    if(ts.step > TRAINING_CONFIG::STEP_LIMIT){
#ifndef LAYER_IN_C_BENCHMARK
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
