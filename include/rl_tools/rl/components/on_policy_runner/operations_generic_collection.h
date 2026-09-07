#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_COLLECTION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_COLLECTION_H

#include "collection.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::ValueState<CRITIC, SPEC>& state){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS) malloc(device, state.state);
    }
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::ValueState<CRITIC, SPEC>& state){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS) free(device, state.state);
    }
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::ValueBuffer<CRITIC, SPEC>& buffer){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS) malloc(device, buffer.bootstrap_state);
        malloc(device, buffer.buffer);
    }
    template <typename DEVICE, typename CRITIC, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::ValueBuffer<CRITIC, SPEC>& buffer){
        if constexpr(SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS) free(device, buffer.bootstrap_state);
        free(device, buffer.buffer);
    }
    template <typename DEVICE, typename DATASET_SPEC, typename CRITIC, typename RNG>
    void evaluate_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, CRITIC& critic, rl::components::on_policy_runner::ValueState<CRITIC, DATASET_SPEC>& state, rl::components::on_policy_runner::ValueBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, typename DATASET_SPEC::TI step_i){
        using SPEC = typename DATASET_SPEC::SPEC;
        if constexpr(SPEC::COLLECT_NEXT_OBSERVATIONS){
            if(step_i == 0){
                reset(device, critic, state.state, rng);
                set_all(device, dataset.all_values, (typename SPEC::T)0);
            }
            else{
                auto reset_column = view(device, dataset.all_reset, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
                auto reset_mask = view_transpose(device, reset_column);
                Mode<mode::sequential::ResetMask<mode::Default<>, mode::sequential::ResetMaskSpecification<decltype(reset_mask)>>> reset_mode;
                reset_mode.mask = reset_mask;
                reset(device, critic, state.state, rng, reset_mode);
            }
            auto observations = view_range(device, dataset.all_observations_privileged, step_i * SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
            auto input = reshape_row_major(device, observations, tensor::Prepend<typename SPEC::OBSERVATION_PRIVILEGED::SHAPE, SPEC::N_ENVIRONMENTS>{});
            auto values = view(device, dataset.all_values, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
            auto output = to_tensor(device, values);
            evaluate_step(device, critic, input, state.state, output, buffer.buffer, rng, Mode<mode::sequential::NoAutoResetMode<mode::Rollout<>>>{});
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename OBSERVATIONS, typename CRITIC, typename RNG>
    void evaluate_bootstrap_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, const OBSERVATIONS& next_observations, CRITIC& critic, rl::components::on_policy_runner::ValueState<CRITIC, DATASET_SPEC>& state, rl::components::on_policy_runner::ValueBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, typename DATASET_SPEC::TI step_i){
        using SPEC = typename DATASET_SPEC::SPEC;
        if constexpr(SPEC::COLLECT_NEXT_OBSERVATIONS){
            // Bootstrap the old episode without advancing its live hidden state or sequence counter.
            copy(device, device, state.state, buffer.bootstrap_state);
            auto input = reshape_row_major(device, next_observations, tensor::Prepend<typename SPEC::OBSERVATION_PRIVILEGED::SHAPE, SPEC::N_ENVIRONMENTS>{});
            auto values = view(device, dataset.bootstrap_values, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, 1>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
            auto output = to_tensor(device, values);
            evaluate_step(device, critic, input, buffer.bootstrap_state, output, buffer.buffer, rng, Mode<mode::sequential::NoAutoResetMode<mode::Rollout<>>>{});
        }
    }
    template <typename DEVICE, typename DATASET_SPEC, typename CRITIC, typename RNG, typename MODE>
    void evaluate_rollout_values(DEVICE& device, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, CRITIC& critic, rl::components::on_policy_runner::ValueBuffer<CRITIC, DATASET_SPEC>& buffer, RNG& rng, const Mode<MODE>&){
        using SPEC = typename DATASET_SPEC::SPEC;
        if constexpr(!SPEC::COLLECT_NEXT_OBSERVATIONS){
            using TI = typename SPEC::TI;
            constexpr TI STEPS = mode::is<MODE, mode::on_policy_runner::Sequential> ? DATASET_SPEC::STEPS_PER_ENV + 1 : 1;
            constexpr TI BATCH_SIZE = mode::is<MODE, mode::on_policy_runner::Sequential> ? SPEC::N_ENVIRONMENTS : DATASET_SPEC::STEPS_TOTAL_ALL;
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
    template <typename DEVICE, typename DS, typename SPEC, typename ACTOR, typename BUFFER, typename RNG>
    void interlude(DEVICE& device, rl::components::on_policy_runner::Dataset<DS>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, rl::components::on_policy_runner::ActorEvaluation<ACTOR, BUFFER>& evaluation, RNG& rng, typename SPEC::TI step_i){
        interlude(device, dataset, runner, buffer, evaluation.actor, evaluation.buffer, rng, step_i);
    }
    template <typename DEVICE, typename DS, typename SPEC, typename ENVIRONMENT, typename ACTOR_EVALUATION, typename VALUE_EVALUATION, typename RNG, typename MODE>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DS>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ENVIRONMENT& environment, ACTOR_EVALUATION& actor, VALUE_EVALUATION& values, RNG& rng, const Mode<MODE>& mode){
        static_assert(utils::typing::is_same_v<typename DS::SPEC, SPEC>, "the dataset must be specified over the runner's specification");
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
        static_assert(mode::is<MODE, mode::on_policy_runner::ActorOnly> != mode::is<MODE, mode::on_policy_runner::ActorCritic>, "select actor-only or actor-critic collection");
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect: runner not initialized");
#endif
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION) reset(device, runner, environment, rng);
        prologue(device, dataset, runner, environment, rng);
        for(typename SPEC::TI step_i = 0; step_i < DS::STEPS_PER_ENV; step_i++){
            if constexpr(mode::is<MODE, mode::on_policy_runner::ActorCritic>) evaluate_values(device, dataset, values.critic, values.state, values.buffer, rng, step_i);
            interlude(device, dataset, runner, buffer, actor, rng, step_i);
            epilogue(device, dataset, runner, buffer, environment, rng, step_i);
            if constexpr(mode::is<MODE, mode::on_policy_runner::ActorCritic>) evaluate_bootstrap_values(device, dataset, buffer.next_observations_privileged, values.critic, values.state, values.buffer, rng, step_i);
        }
        if constexpr(mode::is<MODE, mode::on_policy_runner::ActorCritic>) evaluate_rollout_values(device, dataset, values.critic, values.buffer, rng, mode);
    }
    template <typename DEVICE, typename DS, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_BUFFER, typename RNG>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DS>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ENVIRONMENT& environment, ACTOR& actor, ACTOR_BUFFER& actor_buffer, RNG& rng){
        rl::components::on_policy_runner::ActorEvaluation<ACTOR, ACTOR_BUFFER> evaluation{actor, actor_buffer};
        rl::components::on_policy_runner::NoValueEvaluation values;
        collect(device, dataset, runner, buffer, environment, evaluation, values, rng, Mode<mode::on_policy_runner::ActorOnly<>>{});
    }
    template <typename DEVICE, typename DS, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_BUFFER, typename CRITIC, typename RNG, typename MODE>
    void collect(DEVICE& device, rl::components::on_policy_runner::Dataset<DS>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& buffer, ENVIRONMENT& environment, ACTOR& actor, ACTOR_BUFFER& actor_buffer, CRITIC& critic, rl::components::on_policy_runner::ValueState<CRITIC, DS>& value_state, rl::components::on_policy_runner::ValueBuffer<CRITIC, DS>& value_buffer, RNG& rng, const Mode<MODE>& mode){
        static_assert(mode::is<MODE, mode::on_policy_runner::ActorCritic>);
        rl::components::on_policy_runner::ActorEvaluation<ACTOR, ACTOR_BUFFER> evaluation{actor, actor_buffer};
        rl::components::on_policy_runner::ValueEvaluation<CRITIC, DS> values{critic, value_state, value_buffer};
        collect(device, dataset, runner, buffer, environment, evaluation, values, rng, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
