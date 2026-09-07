#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_COLLECTION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ALGORITHMS_PPO_OPERATIONS_GENERIC_COLLECTION_H

#include "collection.h"
#include "../../../mode/mode.h"
#include "../../components/on_policy_runner/operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void malloc(DEVICE& device, rl::algorithms::ppo::CollectionBuffer<CRITIC, SPEC>& buffer){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS){
            malloc(device, buffer.state);
            malloc(device, buffer.bootstrap_state);
        }
        malloc(device, buffer.buffer);
    }
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void free(DEVICE& device, rl::algorithms::ppo::CollectionBuffer<CRITIC, SPEC>& buffer){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS){
            free(device, buffer.state);
            free(device, buffer.bootstrap_state);
        }
        free(device, buffer.buffer);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename CRITIC, typename RNG>
    void evaluate_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, CRITIC& critic, rl::algorithms::ppo::CollectionBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, typename DATASET_SPEC::TI step_i){
        using SPEC = typename DATASET_SPEC::SPEC;
        if constexpr(SPEC::COLLECT_NEXT_OBSERVATIONS){
            if(step_i == 0){
                reset(device, critic, buffer.state, rng);
                set_all(device, dataset.all_values, (typename SPEC::T)0);
            }
            else{
                auto reset_column = view(device, dataset.all_reset, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
                auto reset_mask = view_transpose(device, reset_column);
                Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(reset_mask)>>> reset_mode;
                reset_mode.mask = reset_mask;
                reset(device, critic, buffer.state, rng, reset_mode);
            }
            auto observations = view_range(device, dataset.all_observations_privileged, step_i * SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto input = reshape_row_major(device, observations, tensor::Prepend<typename SPEC::OBSERVATION_PRIVILEGED::SHAPE, SPEC::N_ENVIRONMENTS>{});
            auto values = view(device, dataset.all_values, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
            auto output = to_tensor(device, values);
            evaluate_step(device, critic, input, buffer.state, output, buffer.buffer, rng, Mode<mode::sequential::NoAutoResetMode<mode::Rollout<>>>{});
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename OBSERVATIONS, typename CRITIC, typename RNG>
    void evaluate_bootstrap_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const OBSERVATIONS& next_observations, CRITIC& critic, rl::algorithms::ppo::CollectionBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, typename DATASET_SPEC::TI step_i){
        using SPEC = typename DATASET_SPEC::SPEC;
        if constexpr(SPEC::COLLECT_NEXT_OBSERVATIONS){
            copy(device, device, buffer.state, buffer.bootstrap_state);
            auto input = reshape_row_major(device, next_observations, tensor::Prepend<typename SPEC::OBSERVATION_PRIVILEGED::SHAPE, SPEC::N_ENVIRONMENTS>{});
            auto values = view(device, dataset.bootstrap_values, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
            auto output = to_tensor(device, values);
            evaluate_step(device, critic, input, buffer.bootstrap_state, output, buffer.buffer, rng, Mode<mode::sequential::NoAutoResetMode<mode::Rollout<>>>{});
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename CRITIC, typename RNG, typename PPO_PARAMETERS>
    void evaluate_rollout_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, CRITIC& critic, rl::algorithms::ppo::CollectionBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, PPO_PARAMETERS){
        using SPEC = typename DATASET_SPEC::SPEC;
        static_assert(SPEC::COLLECT_NEXT_OBSERVATIONS == (PPO_PARAMETERS::BOOTSTRAP_TRUNCATIONS || PPO_PARAMETERS::IGNORE_TERMINATION), "Runner observation capture must match PPO bootstrapping parameters");
        if constexpr(!SPEC::COLLECT_NEXT_OBSERVATIONS){
            using TI = typename SPEC::TI;
            constexpr TI STEPS = PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? DATASET_SPEC::STEPS_PER_ENV + 1 : 1;
            constexpr TI BATCH_SIZE = PPO_PARAMETERS::STATEFUL_ACTOR_AND_CRITIC ? SPEC::N_ENVIRONMENTS : DATASET_SPEC::STEPS_TOTAL_ALL;
            auto input = reshape_row_major(device, dataset.all_observations_privileged, tensor::Prepend<tensor::Prepend<typename SPEC::OBSERVATION_PRIVILEGED::SHAPE, BATCH_SIZE>, STEPS>{});
            auto values = to_tensor(device, dataset.all_values);
            auto output = reshape_row_major(device, values, tensor::Shape<TI, STEPS, BATCH_SIZE, 1>{});
            auto reset_tensor = to_tensor(device, dataset.all_reset);
            auto resets = reshape_row_major(device, reset_tensor, tensor::Shape<TI, STEPS, BATCH_SIZE, 1>{});
            Mode<mode::sequential::ResetMode<mode::Rollout<>, mode::sequential::ResetModeSpecification<TI, decltype(resets)>>> mode;
            mode.reset_container = resets;
            evaluate(device, critic, input, output, buffer.buffer, rng, mode);
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename PPO_SPEC, typename ACTOR_BUFFER, typename RNG>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& runner_buffer, ENVIRONMENT& environment, rl::algorithms::PPO<PPO_SPEC>& ppo, ACTOR_BUFFER& actor_buffer, rl::algorithms::ppo::CollectionBuffer<typename PPO_SPEC::CRITIC_TYPE, DATASET_SPEC>& critic_buffer, RNG& rng){
        static_assert(utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>);
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            reset(device, runner, environment, rng);
        }
        prologue(device, dataset, runner, environment, rng);
        for(typename SPEC::TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            evaluate_values(device, dataset, ppo.critic, critic_buffer, rng, step_i);
            interlude(device, dataset, runner, runner_buffer, ppo.actor, actor_buffer, rng, step_i);
            epilogue(device, dataset, runner, runner_buffer, environment, rng, step_i);
            evaluate_bootstrap_values(device, dataset, runner_buffer.next_observations_privileged, ppo.critic, critic_buffer, rng, step_i);
        }
        evaluate_rollout_values(device, dataset, ppo.critic, critic_buffer, rng, typename PPO_SPEC::PARAMETERS{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
