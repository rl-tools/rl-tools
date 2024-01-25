#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_STATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_SAC_LOOP_CORE_STATE_H

#include "../../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../../rl/components/off_policy_runner/operations_generic.h"

#include "../../../../../rl/utils/evaluation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::algorithms::sac::loop::core{
    // Config State (Init/Step)
    template<typename T_SPEC>
    struct TrainingState{
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
        TI step;
    };
}
#endif