#include "../../../rl/components/off_policy_runner/operations_generic.h"
#include "../../../rl/algorithms/td3/operations_generic.h"

#include "../../../rl/utils/evaluation.h"


BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::algorithms::td3::loop {
    template<typename T_SPEC>
    struct CoreTrainingState {
        using SPEC = T_SPEC;
        using DEVICE = typename SPEC::DEVICE;
        using TI = typename DEVICE::index_t;
        DEVICE device;
        typename SPEC::OPTIMIZER actor_optimizer, critic_optimizers[2];
        decltype(random::default_engine(typename DEVICE::SPEC::RANDOM())) rng, rng_eval;
        typename SPEC::UI ui;
        rl::components::OffPolicyRunner<typename SPEC::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
        typename SPEC::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        typename SPEC::ENVIRONMENT_EVALUATION env_eval;
        typename SPEC::ACTOR_CRITIC_TYPE actor_critic;
        typename SPEC::ACTOR_TYPE::template DoubleBuffer<1> actor_deterministic_evaluation_buffers;
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
        rl::algorithms::td3::CriticTrainingBuffers<typename SPEC::ACTOR_CRITIC_SPEC> critic_training_buffers;
        MatrixDynamic<matrix::Specification<typename SPEC::T, TI, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
        typename SPEC::CRITIC_TYPE::template DoubleBuffer<SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
        rl::algorithms::td3::ActorTrainingBuffers<typename SPEC::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
        typename SPEC::ACTOR_TYPE::template DoubleBuffer<SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
        typename SPEC::ACTOR_TYPE::template DoubleBuffer<SPEC::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    };

    template<typename CONFIG>
    struct TrainingState : CoreTrainingState<CONFIG> {
        using T = typename CONFIG::T;
        using TI = typename CONFIG::DEVICE::index_t;
        TI step = 0;
        bool finished = false;
        static constexpr TI N_EVALUATIONS = CONFIG::STEP_LIMIT / CONFIG::EVALUATION_INTERVAL + 1;
        static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
        rl::utils::evaluation::Result<T, TI, CONFIG::NUM_EVALUATION_EPISODES> evaluation_results[N_EVALUATIONS];
//        T evaluation_results[N_EVALUATIONS];
    };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END


BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::rl::algorithms::td3::loop{
    template <typename TRAINING_STATE>
    void init(TRAINING_STATE& ts, typename TRAINING_STATE::SPEC::DEVICE::index_t seed){
        using SPEC = typename TRAINING_STATE::SPEC;
        using T = typename SPEC::T;

        ts.rng = random::default_engine(typename SPEC::DEVICE::SPEC::RANDOM(), seed);
        ts.rng_eval = random::default_engine(typename SPEC::DEVICE::SPEC::RANDOM(), seed);

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
        ts.finished = false;
        if constexpr(SPEC::CONSTRUCT_LOGGER){
            construct(ts.device, ts.device.logger);
        }
    }

    template <typename TRAINING_STATE>
    void destroy(TRAINING_STATE& ts){
        free(ts.device, ts.actor_critic);
        free(ts.device, ts.off_policy_runner);
        free(ts.device, ts.critic_batch);
        free(ts.device, ts.critic_training_buffers);
        free(ts.device, ts.critic_buffers[0]);
        free(ts.device, ts.critic_buffers[1]);
        free(ts.device, ts.actor_batch);
        free(ts.device, ts.actor_training_buffers);
        free(ts.device, ts.actor_buffers_eval);
        free(ts.device, ts.actor_buffers[0]);
        free(ts.device, ts.actor_buffers[1]);
        free(ts.device, ts.observations_mean);
        free(ts.device, ts.observations_std);
        free(ts.device, ts.actor_deterministic_evaluation_buffers);
    }


    template <typename TRAINING_STATE>
    bool step(TRAINING_STATE& ts){
        bool finished = false;
        using SPEC = typename TRAINING_STATE::SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        if constexpr(SPEC::DETERMINISTIC_EVALUATION == true){
            if(ts.step % SPEC::EVALUATION_INTERVAL == 0){
                auto result = evaluate(ts.device, ts.env_eval, ts.ui, ts.actor_critic.actor, utils::evaluation::Specification<SPEC::NUM_EVALUATION_EPISODES, SPEC::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng_eval, false);
                logging::text(ts.device, ts.device.logger, "Step: ", ts.step, " (mean return: ", result.returns_mean, ", mean episode length: ", result.episode_length_mean, ")");
                TI current_evaluation_i = ts.step / SPEC::EVALUATION_INTERVAL;
                assert(current_evaluation_i < TRAINING_STATE::N_EVALUATIONS);
                ts.evaluation_results[current_evaluation_i] = result;
                add_scalar(ts.device, ts.device.logger, "evaluation/returns_mean", result.returns_mean);
                add_scalar(ts.device, ts.device.logger, "evaluation/returns_std", result.returns_std);
                add_scalar(ts.device, ts.device.logger, "evaluation/episode_length_mean", result.episode_length_mean);
                add_scalar(ts.device, ts.device.logger, "evaluation/episode_length_std", result.episode_length_std);
            }
        }
        step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(ts.step > SPEC::N_WARMUP_STEPS_CRITIC && ts.step % SPEC::TD3_PARAMETERS::CRITIC_TRAINING_INTERVAL == 0){
            for(TI critic_i = 0; critic_i < 2; critic_i++){
                target_action_noise(ts.device, ts.actor_critic, ts.critic_training_buffers.target_next_action_noise, ts.rng);
                gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers);
            }
            T critic_1_loss = critic_loss(ts.device, ts.actor_critic, ts.actor_critic.critic_1, ts.critic_batch, ts.actor_buffers[0], ts.critic_buffers[0], ts.critic_training_buffers);
            add_scalar(ts.device, ts.device.logger, "critic_1_loss", critic_1_loss, 100);
        }

        if(ts.step > SPEC::N_WARMUP_STEPS_ACTOR && ts.step % SPEC::TD3_PARAMETERS::ACTOR_TRAINING_INTERVAL == 0){
            gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
            train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers);

            T actor_value = mean(ts.device, ts.actor_training_buffers.state_action_value);
            add_scalar(ts.device, ts.device.logger, "actor_value", actor_value, 100);
        }
        if(ts.step > SPEC::N_WARMUP_STEPS_CRITIC && ts.step % SPEC::TD3_PARAMETERS::CRITIC_TARGET_UPDATE_INTERVAL == 0){
            update_critic_targets(ts.device, ts.actor_critic);
        }
        if(ts.step > SPEC::N_WARMUP_STEPS_ACTOR && ts.step % SPEC::TD3_PARAMETERS::ACTOR_TARGET_UPDATE_INTERVAL == 0) {
            update_actor_target(ts.device, ts.actor_critic);
        }


        ts.step++;
        ts.finished = ts.step >= SPEC::STEP_LIMIT;
        return ts.finished;
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END


