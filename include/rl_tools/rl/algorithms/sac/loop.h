#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_H

#include "../../../rl/algorithms/sac/operations_generic.h"
#include "../../../rl/components/off_policy_runner/operations_generic.h"


#include "../../../rl/utils/evaluation.h"



RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac::loop{
    template <typename T_SPEC>
    struct CoreTrainingState{
        using SPEC = T_SPEC;
        using DEVICE = typename SPEC::DEVICE;
        using TI = typename DEVICE::index_t;
        DEVICE device;
        typename SPEC::OPTIMIZER actor_optimizer, critic_optimizers[2];
        decltype(random::default_engine(typename DEVICE::SPEC::RANDOM())) rng, rng_eval;
        typename SPEC::UI ui;
        rl::components::OffPolicyRunner<typename SPEC::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
        typename SPEC::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
        typename SPEC::ACTOR_CRITIC_TYPE actor_critic;
        typename SPEC::ACTOR_TYPE::template Buffer<1> actor_deterministic_evaluation_buffers;
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
        rl::algorithms::sac::CriticTrainingBuffers<typename SPEC::ACTOR_CRITIC_SPEC> critic_training_buffers;
        MatrixDynamic<matrix::Specification<typename SPEC::T, TI, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
        typename SPEC::CRITIC_TYPE::template Buffer<SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
        rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
        rl::algorithms::sac::ActorTrainingBuffers<typename SPEC::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
        typename SPEC::ACTOR_TYPE::template Buffer<SPEC::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
        typename SPEC::ACTOR_TYPE::template Buffer<SPEC::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
    };

    template <typename T_SPEC>
    struct TrainingState: CoreTrainingState<T_SPEC>{
        using SPEC = T_SPEC;
        using T = typename T_SPEC::T;
        using TI = typename T_SPEC::DEVICE::index_t;
        TI step = 0;
        static constexpr TI N_EVALUATIONS = T_SPEC::STEP_LIMIT / T_SPEC::EVALUATION_INTERVAL;
        static_assert(N_EVALUATIONS > 0 && N_EVALUATIONS < 1000000);
        rl::utils::evaluation::Result<T, TI, T_SPEC::NUM_EVALUATION_EPISODES> evaluation_results[N_EVALUATIONS];
    };


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
        rl_tools::init(ts.device, ts.envs[0], ts.ui);

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

        ts.off_policy_runner.parameters = rl::components::off_policy_runner::default_parameters<T>;

        construct(ts.device, ts.device.logger);

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
        using SPEC = typename TRAINING_STATE::SPEC;
        step(ts.device, ts.off_policy_runner, ts.actor_critic.actor, ts.actor_buffers_eval, ts.rng);
        if(ts.step > SPEC::N_WARMUP_STEPS){
            for(int critic_i = 0; critic_i < 2; critic_i++){
                gather_batch(ts.device, ts.off_policy_runner, ts.critic_batch, ts.rng);
                train_critic(ts.device, ts.actor_critic, critic_i == 0 ? ts.actor_critic.critic_1 : ts.actor_critic.critic_2, ts.critic_batch, ts.critic_optimizers[critic_i], ts.actor_buffers[critic_i], ts.critic_buffers[critic_i], ts.critic_training_buffers, ts.rng);
            }
            if(ts.step % 1 == 0){
                {
                    gather_batch(ts.device, ts.off_policy_runner, ts.actor_batch, ts.rng);
                    train_actor(ts.device, ts.actor_critic, ts.actor_batch, ts.actor_optimizer, ts.actor_buffers[0], ts.critic_buffers[0], ts.actor_training_buffers, ts.rng);
                }
                update_critic_targets(ts.device, ts.actor_critic);
            }
        }
        if constexpr(SPEC::DETERMINISTIC_EVALUATION == true){
            if(ts.step % SPEC::EVALUATION_INTERVAL == 0){
                auto result = evaluate(ts.device, ts.envs[0], ts.ui, ts.actor_critic.actor, rl::utils::evaluation::Specification<TRAINING_STATE::SPEC::NUM_EVALUATION_EPISODES, SPEC::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng, false);
                std::cout << "Step: " << ts.step << " Mean return: " << result.returns_mean << std::endl;
                ts.evaluation_results[ts.step / SPEC::EVALUATION_INTERVAL] = result;
            }
        }
        ts.step++;
        if(ts.step > SPEC::STEP_LIMIT){
            return true;
        }
        else{
            return finished;
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
