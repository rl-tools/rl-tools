#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_STATE_H

#include "../../../../../rl/algorithms/sac/sac.h"
#include "../../../../../rl/components/off_policy_runner/off_policy_runner.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::sac::loop::core{
        // Config State (Init/Step)
        template<typename T_CONFIG>
        struct State{
            using CONFIG = T_CONFIG;
            using T = typename CONFIG::T;
            using TI = typename CONFIG::TI;
            typename CONFIG::NN::ACTOR_OPTIMIZER actor_optimizer;
            typename CONFIG::NN::CRITIC_OPTIMIZER critic_optimizers[2];
            typename CONFIG::NN::ALPHA_OPTIMIZER alpha_optimizer;
            typename CONFIG::RNG rng;
            rl::components::OffPolicyRunner<typename CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
            typename CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
            typename CONFIG::ENVIRONMENT::Parameters env_parameters[decltype(off_policy_runner)::N_ENVIRONMENTS];
            typename CONFIG::ACTOR_CRITIC_TYPE actor_critic;
            rl::components::off_policy_runner::SequentialBatch<rl::components::off_policy_runner::SequentialBatchSpecification<typename decltype(off_policy_runner)::SPEC, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::SEQUENCE_LENGTH, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE, CONFIG::DYNAMIC_ALLOCATION>> critic_batch;
            rl::algorithms::sac::CriticTrainingBuffers<rl::algorithms::sac::CriticTrainingBuffersSpecification<typename CONFIG::ACTOR_CRITIC_SPEC, CONFIG::DYNAMIC_ALLOCATION>> critic_training_buffers[2];
            Matrix<matrix::Specification<T, TI, CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::SEQUENCE_LENGTH * CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE, CONFIG::ENVIRONMENT::ACTION_DIM, CONFIG::DYNAMIC_ALLOCATION>> action_noise_critic;
            typename CONFIG::NN::CRITIC_TYPE::template Buffer<CONFIG::DYNAMIC_ALLOCATION> critic_buffers[2];
            rl::components::off_policy_runner::SequentialBatch<rl::components::off_policy_runner::SequentialBatchSpecification<typename decltype(off_policy_runner)::SPEC, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::SEQUENCE_LENGTH, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE, CONFIG::DYNAMIC_ALLOCATION>> actor_batch;
            rl::algorithms::sac::ActorTrainingBuffers<rl::algorithms::sac::ActorTrainingBuffersSpecification<typename CONFIG::ACTOR_CRITIC_TYPE::SPEC, CONFIG::DYNAMIC_ALLOCATION>> actor_training_buffers;
            Matrix<matrix::Specification<T, TI, CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::SEQUENCE_LENGTH * CONFIG::CORE_PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE, CONFIG::ENVIRONMENT::ACTION_DIM, CONFIG::DYNAMIC_ALLOCATION>> action_noise_actor;
            typename CONFIG::NN::ACTOR_TYPE::template Buffer<CONFIG::DYNAMIC_ALLOCATION> actor_buffers[2];
            typename CONFIG::NN::ACTOR_TYPE::template Buffer<CONFIG::DYNAMIC_ALLOCATION> actor_buffers_eval;
            TI step;
            bool allocated = false;
            bool warmup_policy_transitioned = false;
        };
    }
    template <typename T_CONFIG>
    constexpr auto& get_actor(rl::algorithms::sac::loop::core::State<T_CONFIG>& ts){
        return ts.actor_critic.actor;
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif