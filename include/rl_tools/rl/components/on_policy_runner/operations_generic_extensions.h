#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_EXTENSIONS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_GENERIC_EXTENSIONS_H

#include "operations_generic.h"
#include "collection.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& buffer){
        malloc(device, buffer.observations);
        malloc(device, buffer.actions);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& buffer){
        free(device, buffer.observations);
        free(device, buffer.actions);
    }
    template <typename DEVICE, typename DEVICE_EVALUATION, typename DS, typename SPEC, typename ACTOR, typename ACTOR_EVAL, typename ACTOR_BUFFER, typename RNG, typename RNG_EVAL>
    void interlude(DEVICE& device, DEVICE_EVALUATION& evaluation_device, rl::components::on_policy_runner::Dataset<DS>& dataset, rl::components::on_policy_runner::Buffer<SPEC>& runner_buffer, ACTOR& actor, ACTOR_EVAL& evaluation_actor, ACTOR_BUFFER& actor_buffer, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& transfer, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& evaluation_transfer, RNG& rng, RNG_EVAL& evaluation_rng, typename SPEC::TI step_i){
        auto observations = view_range(device, dataset.all_observations, step_i * SPEC::N_ENVIRONMENTS, tensor::ViewSpec<0, SPEC::N_ENVIRONMENTS>{});
        auto observations_matrix = matrix_view(device, observations);
        copy(device, evaluation_device, observations_matrix, evaluation_transfer.observations);
        auto input_tensor = to_tensor(evaluation_device, evaluation_transfer.observations);
        auto input = reshape_row_major(evaluation_device, input_tensor, tensor::Prepend<tensor::Prepend<typename SPEC::OBSERVATION::SHAPE, SPEC::N_ENVIRONMENTS>, 1>{});
        auto output_tensor = to_tensor(evaluation_device, evaluation_transfer.actions);
        auto output = unsqueeze(evaluation_device, output_tensor);
        evaluate(evaluation_device, evaluation_actor, input, output, actor_buffer, evaluation_rng, Mode<mode::Rollout<>>{});
        copy(evaluation_device, device, evaluation_transfer.actions, transfer.actions);
        auto actions_mean = view(device, dataset.actions_mean, matrix::ViewSpec<SPEC::N_ENVIRONMENTS, SPEC::BATCH_ENVIRONMENT::ACTION_DIM>{}, step_i * SPEC::N_ENVIRONMENTS, 0);
        copy(device, device, transfer.actions, actions_mean);
        auto& last_layer = get_last_layer(actor);
        auto log_std = matrix_view(device, last_layer.log_std.parameters);
        sample_actions(device, dataset, log_std, runner_buffer.actions, step_i, rng);
    }
    template <typename DEVICE, typename DEVICE_EVALUATION, typename DATASET_SPEC, typename SPEC, typename ENVIRONMENT, typename ACTOR, typename ACTOR_EVALUATION, typename POLICY_EVAL_BUFFERS, typename RNG, typename RNG_EVALUATION>
    void collect_hybrid(DEVICE& device, DEVICE_EVALUATION& device_evaluation, rl::components::on_policy_runner::Dataset<DATASET_SPEC>& dataset, rl::components::OnPolicyRunner<SPEC>& runner, rl::components::on_policy_runner::Buffer<SPEC>& runner_buffer, ENVIRONMENT& environment, ACTOR& actor, ACTOR_EVALUATION& actor_evaluation, POLICY_EVAL_BUFFERS& policy_eval_buffers, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& evaluation_buffer, rl::components::on_policy_runner::CollectionEvaluationBuffer<SPEC>& evaluation_buffer_evaluation, RNG& rng, RNG_EVALUATION& rng_evaluation){
        static_assert(utils::typing::is_same_v<typename DATASET_SPEC::SPEC, SPEC>, "the dataset must be specified over the runner's specification");
        static_assert(utils::typing::is_same_v<typename SPEC::BATCH_ENVIRONMENT, ENVIRONMENT>, "the runner and environment types must match");
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_ON_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "rl::components::on_policy_runner::collect_hybrid: runner not initialized");
#endif
        using TI = typename SPEC::TI;
        if constexpr(SPEC::TRUNCATE_ON_EACH_ITERATION){
            reset(device, runner, environment, rng);
        }
        prologue(device, dataset, runner, environment, rng);
        for(TI step_i = 0; step_i < DATASET_SPEC::STEPS_PER_ENV; step_i++){
            interlude(device, device_evaluation, dataset, runner_buffer, actor, actor_evaluation, policy_eval_buffers, evaluation_buffer, evaluation_buffer_evaluation, rng, rng_evaluation, step_i);
            epilogue(device, dataset, runner, runner_buffer, environment, rng, step_i);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
