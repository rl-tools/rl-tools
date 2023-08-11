#include <backprop_tools/rl/components/off_policy_runner/operations_generic.h>
#include <backprop_tools/rl/algorithms/td3/operations_generic.h>

#include <backprop_tools/rl/utils/evaluation.h>


namespace backprop_tools::rl::algorithms::td3::loop {
    template<typename T_TRAINING_CONFIG>
    struct CoreTrainingState {
        using TRAINING_CONFIG = T_TRAINING_CONFIG;
        using DEVICE = typename TRAINING_CONFIG::DEVICE;
        using TI = typename DEVICE::index_t;
        typename DEVICE::SPEC::LOGGING logger;
        DEVICE device;
        typename TRAINING_CONFIG::OPTIMIZER actor_optimizer, critic_optimizers[2];
        decltype(random::default_engine(typename DEVICE::SPEC::RANDOM())) rng;
        typename TRAINING_CONFIG::UI ui;
        rl::components::OffPolicyRunner<typename TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
        typename TRAINING_CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE actor_critic;
        typename TRAINING_CONFIG::ACTOR_TYPE::template DoubleBuffer<1> actor_deterministic_evaluation_buffers;
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
        rl::algorithms::td3::CriticTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_SPEC> critic_training_buffers;
        MatrixDynamic<matrix::Specification<typename TRAINING_CONFIG::T, TI, 1, TRAINING_CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
        typename TRAINING_CONFIG::CRITIC_TYPE::template DoubleBuffer<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
        rl::algorithms::td3::ActorTrainingBuffers<typename TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
        typename TRAINING_CONFIG::ACTOR_TYPE::template DoubleBuffer<TRAINING_CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
        typename TRAINING_CONFIG::ACTOR_TYPE::template DoubleBuffer<TRAINING_CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    };

    template<typename TRAINING_CONFIG>
    struct TrainingState : CoreTrainingState<TRAINING_CONFIG> {
        using T = typename TRAINING_CONFIG::T;
        using TI = typename TRAINING_CONFIG::DEVICE::index_t;
        TI step = 0;
        static constexpr TI N_EVALUATIONS = TRAINING_CONFIG::STEP_LIMIT / TRAINING_CONFIG::EVALUATION_INTERVAL;
        static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
        T evaluation_returns[N_EVALUATIONS];
    };
}


namespace backprop_tools::rl::algorithms::td3::loop{
    template <typename TRAINING_STATE>
    void init(TRAINING_STATE& ts, typename TRAINING_STATE::TRAINING_CONFIG::DEVICE::index_t seed){
        using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;
        using T = typename TRAINING_CONFIG::T;

        ts.device.logger = &ts.logger;

        ts.rng = random::default_engine(typename TRAINING_CONFIG::DEVICE::SPEC::RANDOM(), seed);

        malloc(ts.device, ts.actor_critic);
        init(ts.device, ts.actor_critic, ts.rng);

        malloc(ts.device, ts.off_policy_runner);
        init(ts.device, ts.off_policy_runner, ts.envs);
        backprop_tools::init(ts.device, ts.envs[0], ts.ui);

        malloc(ts.device, ts.critic_batch);
        malloc(ts.device, ts.critic_training_buffers);
        malloc(ts.device, ts.critic_buffers[0]);
        malloc(ts.device, ts.critic_buffers[1]);

        malloc(ts.device, ts.actor_batch);
        malloc(ts.device, ts.actor_training_buffers);
        malloc(ts.device, ts.actor_buffers_eval);
        malloc(ts.device, ts.actor_buffers[0]);
        malloc(ts.device, ts.actor_buffers[1]);

        malloc(ts.device, ts.observations_mean);
        malloc(ts.device, ts.observations_std);

        malloc(ts.device, ts.actor_deterministic_evaluation_buffers);

        set_all(ts.device, ts.observations_mean, 0);
        set_all(ts.device, ts.observations_std, 1);

        ts.off_policy_runner.parameters = components::off_policy_runner::default_parameters<T>;

        ts.step = 0;
    }

    template <typename TRAINING_STATE>
    void destroy(TRAINING_STATE& ts){
        free(ts.device, ts.critic_batch);
        free(ts.device, ts.critic_training_buffers);
        free(ts.device, ts.actor_batch);
        free(ts.device, ts.actor_training_buffers);
        free(ts.device, ts.off_policy_runner);
        free(ts.device, ts.actor_critic);
        free(ts.device, ts.observations_mean);
        free(ts.device, ts.observations_std);
    }


    template <typename TRAINING_STATE>
    bool step(TRAINING_STATE& ts){
        bool finished = false;
        using TRAINING_CONFIG = typename TRAINING_STATE::TRAINING_CONFIG;
        step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(ts.step > TRAINING_CONFIG::N_WARMUP_STEPS){

            for(int critic_i = 0; critic_i < 2; critic_i++){
                target_action_noise(ts.device, ts.actor_critic, ts.critic_training_buffers.target_next_action_noise, ts.rng);
                gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
            }

            if(ts.step % 2 == 0){
                {
                    gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                    train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers);
                }

                update_critic_targets(ts.device, ts.actor_critic);
                update_actor_target(ts.device, ts.actor_critic);
            }
        }
        if constexpr(TRAINING_CONFIG::DETERMINISTIC_EVALUATION == true){
            if(ts.step % TRAINING_CONFIG::EVALUATION_INTERVAL == 0){
                auto result = evaluate(ts.device, ts.envs[0], ts.ui, ts.actor_critic.actor, utils::evaluation::Specification<1, TRAINING_CONFIG::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng, false);
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
}


