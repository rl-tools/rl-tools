#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include "../../../math/operations_generic.h"
#include "off_policy_runner.h"

#include "../../../rl/components/replay_buffer/operations_generic.h"

#include "operations_generic_per_env.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        malloc(device, buffers.observations);
        malloc(device, buffers.actions);
        malloc(device, buffers.next_observations);

        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            malloc(device, buffers.observations_privileged);
            malloc(device, buffers.next_observations_privileged);
        }
        else{
            buffers.observations_privileged = buffers.observations;
            buffers.next_observations_privileged = buffers.next_observations;
        }
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::EpisodeStats<SPEC>& episode_stats) {
        malloc(device, episode_stats.data);
        episode_stats.returns = view<DEVICE, typename decltype(episode_stats.data)::SPEC, SPEC::EPISODE_STATS_BUFFER_SIZE, 1>(device, episode_stats.data, 0, 0);
        episode_stats.steps   = view<DEVICE, typename decltype(episode_stats.data)::SPEC, SPEC::EPISODE_STATS_BUFFER_SIZE, 1>(device, episode_stats.data, 0, 1);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        malloc(device, runner.buffers);
        malloc(device, runner.states);
        malloc(device, runner.episode_return);
        malloc(device, runner.episode_step);
        malloc(device, runner.truncated);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            malloc(device, runner.replay_buffers[env_i]);
            malloc(device, runner.episode_stats[env_i]);
        }
    }
    template <typename DEVICE, typename BATCH_SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch) {
        using BATCH = rl::components::off_policy_runner::Batch<BATCH_SPEC>;
        using SPEC = typename BATCH_SPEC::SPEC;
        using DATA_SPEC = typename decltype(batch.observations_actions_next_observations)::SPEC;
        constexpr typename DEVICE::index_t BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        malloc(device, batch.observations_actions_next_observations);
        typename DEVICE::index_t offset = 0;
        batch.observations                 = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                               >(device, batch.observations_actions_next_observations, 0, offset); offset += BATCH::ASYMMETRIC_OBSERVATIONS ? BATCH::OBSERVATION_DIM : 0;
        batch.observations_and_actions     = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM_PRIVILEGED + BATCH::ACTION_DIM>(device, batch.observations_actions_next_observations, 0, offset);
        batch.observations_privileged      = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM_PRIVILEGED                    >(device, batch.observations_actions_next_observations, 0, offset); offset += BATCH::OBSERVATION_DIM_PRIVILEGED;
        batch.actions                      = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::     ACTION_DIM                               >(device, batch.observations_actions_next_observations, 0, offset); offset += BATCH::ACTION_DIM;
        batch.next_observations            = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                               >(device, batch.observations_actions_next_observations, 0, offset); offset += BATCH::ASYMMETRIC_OBSERVATIONS ? BATCH::OBSERVATION_DIM : 0;
        batch.next_observations_privileged = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM_PRIVILEGED                    >(device, batch.observations_actions_next_observations, 0, offset); offset += BATCH::OBSERVATION_DIM_PRIVILEGED;

        malloc(device, batch.rewards);
        malloc(device, batch.terminated);
        malloc(device, batch.truncated);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        free(device, buffers.observations);
        free(device, buffers.actions);
        free(device, buffers.next_observations);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::EpisodeStats<SPEC>& episode_stats) {
        free(device, episode_stats.data);
        episode_stats.returns._data = nullptr;
        episode_stats.steps._data = nullptr;
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        free(device, runner.buffers);
        free(device, runner.states);
        free(device, runner.episode_return);
        free(device, runner.episode_step);
        free(device, runner.truncated);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            free(device, runner.replay_buffers[env_i]);
            free(device, runner.episode_stats[env_i]);
        }
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::Batch<SPEC>& batch) {
        free(device, batch.observations_actions_next_observations);
        batch.observations.                _data = nullptr;
        batch.observations_privileged.     _data = nullptr;
        batch.observations_and_actions.    _data = nullptr;
        batch.actions.                     _data = nullptr;
        batch.next_observations.           _data = nullptr;
        batch.next_observations_privileged._data = nullptr;
        free(device, batch.rewards);
        free(device, batch.terminated);
        free(device, batch.truncated);
    }
    template<typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, typename SPEC::ENVIRONMENT envs[SPEC::N_ENVIRONMENTS]) {
        set_all(device, runner.truncated, true);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            init(device, runner.replay_buffers[env_i]);
            runner.envs[env_i] = envs[env_i];
        }
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::off_policy_runner{
        template<typename DEVICE, typename SPEC, typename RNG>
        void prologue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
            using TI = typename DEVICE::index_t;
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
                prologue_per_env(device, runner, rng, env_i);
            }
        }
        template<typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS, typename RNG>
        void interlude(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY &policy, POLICY_BUFFERS& policy_eval_buffers, RNG& rng){
            using TI = typename DEVICE::index_t;
            constexpr TI BATCH_SIZE = decltype(runner.buffers.actions)::ROWS;
            auto action_view = view(device, runner.buffers.actions, matrix::ViewSpec<BATCH_SIZE, POLICY::OUTPUT_DIM>{});
            evaluate(device, policy, runner.buffers.observations, action_view, policy_eval_buffers, rng);
        }

        template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
        void epilogue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, const POLICY& policy, RNG& rng){
            using TI = typename DEVICE::index_t;
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                epilogue_per_env(device, runner, policy, rng, env_i);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS, typename RNG>
    void step(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY& policy, POLICY_BUFFERS& policy_eval_buffers, RNG &rng){
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "OffPolicyRunner not initialized");
#endif
        static_assert(POLICY::INPUT_DIM == SPEC::ENVIRONMENT::OBSERVATION_DIM, "The policy's input dimension must match the environment's observation dimension.");
//        static_assert(POLICY::OUTPUT_DIM == (SPEC::ENVIRONMENT::ACTION_DIM * (SPEC::STOCHASTIC_POLICY ? 2 : 1)), "The policy's output dimension must match the environment's action dimension.");
        static_assert(POLICY::OUTPUT_DIM == SPEC::ENVIRONMENT::ACTION_DIM ||  POLICY::OUTPUT_DIM == 2*SPEC::ENVIRONMENT::ACTION_DIM, "The policy's output dimension must match the environment's action dimension.");
        // todo: increase efficiency by removing the double observation of each state
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;

        rl::components::off_policy_runner::prologue(device, runner, rng);
        rl::components::off_policy_runner::interlude(device, runner, policy, policy_eval_buffers, rng);
        rl::components::off_policy_runner::epilogue(device, runner, policy, rng);
    }
    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC = false>
    void gather_batch(DEVICE& device, const rl::components::ReplayBuffer<SPEC>& replay_buffer, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, typename DEVICE::index_t batch_step_i, RNG& rng) {
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION
        utils::assert_exit(device, replay_buffer.position > 0 || replay_buffer.full, "Replay buffer is empty");
#endif
        typename DEVICE::index_t sample_index_max = (replay_buffer.full ? SPEC::CAPACITY : replay_buffer.position) - 1;
        typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : random::uniform_int_distribution( typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, sample_index_max, rng);

        auto observation_target = row(device, batch.observations, batch_step_i);
        auto observation_source = row(device, replay_buffer.observations, sample_index);
        copy(device, device, observation_source, observation_target);

        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            auto observation_privileged_target = row(device, batch.observations_privileged        , batch_step_i);
            auto observation_privileged_source = row(device, replay_buffer.observations_privileged, sample_index);
            copy(device, device, observation_privileged_source, observation_privileged_target);
        }

        auto action_target = row(device, batch.actions, batch_step_i);
        auto action_source = row(device, replay_buffer.actions, sample_index);
        copy(device, device, action_source, action_target);

        auto next_observation_target = row(device, batch.next_observations, batch_step_i);
        auto next_observation_source = row(device, replay_buffer.next_observations, sample_index);
        copy(device, device, next_observation_source, next_observation_target);

        if constexpr(SPEC::ASYMMETRIC_OBSERVATIONS){
            auto next_observation_privileged_target = row(device, batch.next_observations_privileged        , batch_step_i);
            auto next_observation_privileged_source = row(device, replay_buffer.next_observations_privileged, sample_index);
            copy(device, device, next_observation_privileged_source, next_observation_privileged_target);
        }

        set(batch.rewards, 0, batch_step_i, get(replay_buffer.rewards, sample_index, 0));
        set(batch.terminated, 0, batch_step_i, get(replay_buffer.terminated, sample_index, 0));
        set(batch.truncated, 0, batch_step_i, get(replay_buffer.truncated,  sample_index, 0));
    }
    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC=false>
    void gather_batch(DEVICE& device, const rl::components::OffPolicyRunner<SPEC>& runner, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, RNG& rng) {
        static_assert(utils::typing::is_same_v<SPEC, typename BATCH_SPEC::SPEC>);
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using RUNNER = rl::components::OffPolicyRunner<SPEC>;
        constexpr typename DEVICE::index_t BATCH_SIZE = BATCH_SPEC::BATCH_SIZE;
        for(typename DEVICE::index_t batch_step_i=0; batch_step_i < BATCH_SIZE; batch_step_i++) {
            typename DEVICE::index_t env_i = DETERMINISTIC ? 0 : random::uniform_int_distribution(typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, SPEC::N_ENVIRONMENTS - 1, rng);
            auto& replay_buffer = runner.replay_buffers[env_i];
            gather_batch<DEVICE, typename RUNNER::REPLAY_BUFFER_SPEC, BATCH_SPEC, RNG, DETERMINISTIC>(device, replay_buffer, batch, batch_step_i, rng);
        }
    }
//    template<typename SOURCE_DEVICE, typename TARGET_DEVICE,  typename SOURCE_SPEC, typename TARGET_SPEC>
//    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const  rl::components::off_policy_runner::Batch<SOURCE_SPEC>& source, nn_models::mlp::NeuralNetworkBuffersForwardBackward<TARGET_SPEC>& target){
//        copy(source_device, target_device, (nn_models::mlp::NeuralNetworkBuffers<SOURCE_SPEC>&)source, (nn_models::mlp::NeuralNetworkBuffers<TARGET_SPEC>&)target);
//        copy(source_device, target_device, source.d_input, target.d_input);
//        copy(source_device, target_device, source.d_output, target.d_output);
//    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::components::off_policy_runner::Batch<SOURCE_SPEC>& source, rl::components::off_policy_runner::Batch<TARGET_SPEC>& target){
        copy(source_device, target_device, source.observations_actions_next_observations, target.observations_actions_next_observations);
        copy(source_device, target_device, source.rewards, target.rewards);
        copy(source_device, target_device, source.terminated, target.terminated);
        copy(source_device, target_device, source.truncated, target.truncated);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::components::off_policy_runner::Buffers<SOURCE_SPEC>& source, rl::components::off_policy_runner::Buffers<TARGET_SPEC>& target){
        copy(source_device, target_device, source.observations, target.observations);
        copy(source_device, target_device, source.actions, target.actions);
        copy(source_device, target_device, source.next_observations, target.next_observations);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::components::off_policy_runner::EpisodeStats<SOURCE_SPEC>& source, rl::components::off_policy_runner::EpisodeStats<TARGET_SPEC>& target){
        copy(source_device, target_device, source.data, target.data);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, rl::components::OffPolicyRunner<SOURCE_SPEC>& source, rl::components::OffPolicyRunner<TARGET_SPEC>& target){
        copy(source_device, target_device, source.buffers, target.buffers);
        copy(source_device, target_device, source.states, target.states);
        copy(source_device, target_device, source.episode_return, target.episode_return);
        copy(source_device, target_device, source.episode_step, target.episode_step);
        copy(source_device, target_device, source.truncated, target.truncated);
        for (typename SOURCE_DEVICE::index_t env_i = 0; env_i < SOURCE_SPEC::N_ENVIRONMENTS; env_i++){
            copy(source_device, target_device, source.replay_buffers[env_i], target.replay_buffers[env_i]);
            copy(source_device, target_device, source.episode_stats[env_i], target.episode_stats[env_i]);
            target.envs[env_i] = source.envs[env_i];
        }
#ifdef RL_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        target.initialized = source.initialized;
#endif
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
