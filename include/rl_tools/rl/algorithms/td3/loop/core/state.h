#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_TD3_LOOP_CORE_STATE_H

#include "../../../../../rl/algorithms/td3/td3.h"
#include "../../../../../rl/components/off_policy_runner/off_policy_runner.h"
#include "../../../../../rl/utils/evaluation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::algorithms::td3::loop::core{
        // Config State (Init/Step)
        template<typename T_CONFIG>
        struct State{
            using CONFIG = T_CONFIG;
            using TI = typename CONFIG::TI;
            typename CONFIG::NN::OPTIMIZER actor_optimizer, critic_optimizers[2];
            typename CONFIG::RNG rng;
            rl::components::OffPolicyRunner<typename CONFIG::OFF_POLICY_RUNNER_SPEC> off_policy_runner;
            typename CONFIG::ENVIRONMENT envs[decltype(off_policy_runner)::N_ENVIRONMENTS];
            typename CONFIG::ACTOR_CRITIC_TYPE actor_critic;
            typename CONFIG::NN::ACTOR_TYPE::template Buffer<1> actor_deterministic_evaluation_buffers;
            rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE>> critic_batch;
            rl::algorithms::td3::CriticTrainingBuffers<typename CONFIG::ACTOR_CRITIC_SPEC> critic_training_buffers;
            MatrixDynamic<matrix::Specification<typename CONFIG::T, TI, 1, CONFIG::ENVIRONMENT::OBSERVATION_DIM>> observations_mean, observations_std;
            typename CONFIG::NN::CRITIC_TYPE::template Buffer<CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::CRITIC_BATCH_SIZE> critic_buffers[2];
            rl::components::off_policy_runner::Batch<rl::components::off_policy_runner::BatchSpecification<typename decltype(off_policy_runner)::SPEC, CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE>> actor_batch;
            rl::algorithms::td3::ActorTrainingBuffers<typename CONFIG::ACTOR_CRITIC_TYPE::SPEC> actor_training_buffers;
            typename CONFIG::NN::ACTOR_TYPE::template Buffer<CONFIG::ACTOR_CRITIC_TYPE::SPEC::PARAMETERS::ACTOR_BATCH_SIZE> actor_buffers[2];
            typename CONFIG::NN::ACTOR_TYPE::template Buffer<CONFIG::OFF_POLICY_RUNNER_SPEC::N_ENVIRONMENTS> actor_buffers_eval;
            TI step;
        };
    }
    template <typename T_CONFIG>
    constexpr auto& get_actor(rl::algorithms::td3::loop::core::State<T_CONFIG>& ts){
        return ts.actor_critic.actor;
    }

}
#endif