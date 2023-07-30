#include <backprop_tools/rl/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>




template <typename T_TRAINING_CONFIG>
struct CoreTrainingState{
    using TRAINING_CONFIG = T_TRAINING_CONFIG;
    using DEVICE = typename TRAINING_CONFIG::DEVICE;
    using TI = typename DEVICE::index_t;
    typename DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    typename TRAINING_CONFIG::OPTIMIZER actor_optimizer, critic_optimizers[2];
    decltype(bpt::random::default_engine(typename DEVICE::SPEC::RANDOM())) rng;
    bool ui = false;
    bpt::rl::components::OffPolicyRunner<typename TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
    typename TRAINING_CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
    typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE actor_critic;
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffers<1> actor_deterministic_evaluation_buffers;
    bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
    bpt::rl::algorithms::td3::CriticTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_SPEC> critic_training_buffers;
    bpt::MatrixDynamic<bpt::matrix::Specification<typename TRAINING_CONFIG::T, TI, 1, TRAINING_CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
    typename TRAINING_CONFIG::CRITIC_NETWORK_TYPE::template Buffers<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
    bpt::rl::components::off_policy_runner::Batch<bpt::rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
    bpt::rl::algorithms::td3::ActorTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffers<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
    typename TRAINING_CONFIG::ACTOR_NETWORK_TYPE::template Buffers<TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
};

template <typename TRAINING_CONFIG>
struct TrainingState: CoreTrainingState<TRAINING_CONFIG>{
    using T = typename TRAINING_CONFIG::T;
    using TI = typename TRAINING_CONFIG::DEVICE::index_t;
    TI step = 0;
    static constexpr TI N_EVALUATIONS = TRAINING_CONFIG::STEP_LIMIT / TRAINING_CONFIG::EVALUATION_INTERVAL;
    static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
    T evaluation_returns[N_EVALUATIONS];
};


template <typename TRAINING_STATE>
void training_init(TRAINING_STATE& ts, typename TRAINING_STATE::TRAINING_CONFIG::DEVICE::index_t seed){
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;

    ts.device.logger = &ts.logger;

    ts.rng = bpt::random::default_engine(typename TRAINING_CONFIG::DEVICE::SPEC::RANDOM(), seed);

    bpt::malloc(ts.device, ts.actor_critic);
    bpt::init(ts.device, ts.actor_critic, ts.rng);

    bpt::malloc(ts.device, ts.off_policy_runner);
    bpt::init(ts.device, ts.off_policy_runner, ts.envs);

    bpt::malloc(ts.device, ts.critic_batch);
    bpt::malloc(ts.device, ts.critic_training_buffers);
    bpt::malloc(ts.device, ts.critic_buffers[0]);
    bpt::malloc(ts.device, ts.critic_buffers[1]);

    bpt::malloc(ts.device, ts.actor_batch);
    bpt::malloc(ts.device, ts.actor_training_buffers);
    bpt::malloc(ts.device, ts.actor_buffers_eval);
    bpt::malloc(ts.device, ts.actor_buffers[0]);
    bpt::malloc(ts.device, ts.actor_buffers[1]);

    bpt::malloc(ts.device, ts.observations_mean);
    bpt::malloc(ts.device, ts.observations_std);

    bpt::malloc(ts.device, ts.actor_deterministic_evaluation_buffers);

    bpt::set_all(ts.device, ts.observations_mean, 0);
    bpt::set_all(ts.device, ts.observations_std, 1);

    ts.step = 0;
}

template <typename TRAINING_STATE>
void training_destroy(TRAINING_STATE& ts){
    bpt::free(ts.device, ts.critic_batch);
    bpt::free(ts.device, ts.critic_training_buffers);
    bpt::free(ts.device, ts.actor_batch);
    bpt::free(ts.device, ts.actor_training_buffers);
    bpt::free(ts.device, ts.off_policy_runner);
    bpt::free(ts.device, ts.actor_critic);
    bpt::free(ts.device, ts.observations_mean);
    bpt::free(ts.device, ts.observations_std);
}


template <typename TRAINING_STATE>
bool training_step(TRAINING_STATE& ts){
    bool finished = false;
    using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;
    bpt::step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
    if(ts.step > TRAINING_CONFIG::N_WARMUP_STEPS){

        for(int critic_i = 0; critic_i < 2; critic_i++){
            bpt::target_action_noise(ts.device, ts.actor_critic, ts.critic_training_buffers.target_next_action_noise, ts.rng);
            bpt::gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
            bpt::train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
        }

        if(ts.step % 2 == 0){
            {
                bpt::gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                bpt::train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers);
            }

            bpt::update_critic_targets(ts.device, ts.actor_critic);
            bpt::update_actor_target(ts.device, ts.actor_critic);
        }
    }
    if constexpr(TRAINING_CONFIG::DETERMINISTIC_EVALUATION == true){
        if(ts.step % TRAINING_CONFIG::EVALUATION_INTERVAL == 0){
            auto result = bpt::evaluate(ts.device, ts.envs[0], ts.ui, ts.actor_critic.actor, bpt::rl::utils::evaluation::Specification<1, TRAINING_CONFIG::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng, false);
            std::cout << "Step: " << ts.step << " Mean return: " << result.returns_mean << std::endl;
            ts.evaluation_returns[ts.step / TRAINING_CONFIG::EVALUATION_INTERVAL] = result.returns_mean;
        }
    }
    ts.step++;
    if(ts.step > TRAINING_CONFIG::STEP_LIMIT){
        return true;
    }
    else{
        return finished;
    }
}
