#ifndef BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define BACKPROP_TOOLS_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include <backprop_tools/math/operations_generic.h>
#include "off_policy_runner.h"

#include <backprop_tools/rl/components/replay_buffer/operations_generic.h>

#include "operations_generic_per_env.h"

namespace backprop_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        malloc(device, buffers.observations);
        malloc(device, buffers.actions);
        malloc(device, buffers.next_observations);
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
        batch.observations             = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                    >(device, batch.observations_actions_next_observations, 0, 0);
        batch.actions                  = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::     ACTION_DIM                    >(device, batch.observations_actions_next_observations, 0, BATCH::OBSERVATION_DIM);
        batch.next_observations        = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM                    >(device, batch.observations_actions_next_observations, 0, BATCH::OBSERVATION_DIM + BATCH::ACTION_DIM);
        batch.observations_and_actions = view<DEVICE, DATA_SPEC, BATCH_SIZE, BATCH::OBSERVATION_DIM + BATCH::ACTION_DIM>(device, batch.observations_actions_next_observations, 0, 0);

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
        batch.observations.            _data = nullptr;
        batch.actions.                 _data = nullptr;
        batch.next_observations.       _data = nullptr;
        batch.observations_and_actions._data = nullptr;
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
#ifdef BACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        runner.initialized = true;
#endif
    }
    namespace rl::components::off_policy_runner{
        template<typename DEVICE, typename SPEC, typename RNG>
        void prologue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
            using TI = typename DEVICE::index_t;
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
                prologue_per_env(device, &runner, rng, env_i);
            }
        }
        template<typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS>
        void interlude(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY &policy, POLICY_BUFFERS& policy_eval_buffers) {
            evaluate(device, policy, runner.buffers.observations, runner.buffers.actions, policy_eval_buffers);
        }

        template<typename DEVICE, typename SPEC, typename RNG>
        void epilogue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, RNG &rng) {
            using TI = typename DEVICE::index_t;
            for (TI env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
                epilogue_per_env(device, &runner, rng, env_i);
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename POLICY, typename POLICY_BUFFERS, typename RNG>
    void step(DEVICE& device, rl::components::OffPolicyRunner<SPEC>& runner, POLICY& policy, POLICY_BUFFERS& policy_eval_buffers, RNG &rng) {
#ifdef BACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        utils::assert_exit(device, runner.initialized, "OffPolicyRunner not initialized");
#endif
        static_assert(POLICY::INPUT_DIM == SPEC::ENVIRONMENT::OBSERVATION_DIM,
                      "The policy's input dimension must match the environment's observation dimension.");
        static_assert(POLICY::OUTPUT_DIM == SPEC::ENVIRONMENT::ACTION_DIM,
                      "The policy's output dimension must match the environment's action dimension.");
        // todo: increase efficiency by removing the double observation of each state
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;

        rl::components::off_policy_runner::prologue(device, runner, rng);
        rl::components::off_policy_runner::interlude(device, runner, policy, policy_eval_buffers);
        rl::components::off_policy_runner::epilogue(device, runner, rng);
    }
    template <typename DEVICE, typename SPEC, typename BATCH_SPEC, typename RNG, bool DETERMINISTIC = false>
    void gather_batch(DEVICE& device, const rl::components::ReplayBuffer<SPEC>& replay_buffer, rl::components::off_policy_runner::Batch<BATCH_SPEC>& batch, typename DEVICE::index_t batch_step_i, RNG& rng) {
#ifdef BACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_GATHER_BATCH_CHECK_REPLAY_BUFFER_POSITION
        utils::assert_exit(device, replay_buffer.position > 0, "Replay buffer is empty");
#endif
        typename DEVICE::index_t sample_index_max = (replay_buffer.full ? SPEC::CAPACITY : replay_buffer.position) - 1;
        typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : random::uniform_int_distribution( typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, sample_index_max, rng);

        auto observation_target = view<DEVICE, typename decltype(batch.observations)::SPEC, 1, SPEC::OBSERVATION_DIM>(device, batch.observations, batch_step_i, 0);
        auto observation_source = view<DEVICE, typename decltype(replay_buffer.observations)::SPEC, 1, SPEC::OBSERVATION_DIM>(device, replay_buffer.observations, sample_index, 0);
        copy(device, device, observation_target, observation_source);

        auto action_target = view<DEVICE, typename decltype(batch.actions)::SPEC, 1, SPEC::ACTION_DIM>(device, batch.actions, batch_step_i, 0);
        auto action_source = view<DEVICE, typename decltype(replay_buffer.actions)::SPEC, 1, SPEC::ACTION_DIM>(device, replay_buffer.actions, sample_index, 0);
        copy(device, device, action_target, action_source);

        auto next_observation_target = view<DEVICE, typename decltype(batch.next_observations)::SPEC, 1, SPEC::OBSERVATION_DIM>(device, batch.next_observations, batch_step_i, 0);
        auto next_observation_source = view<DEVICE, typename decltype(replay_buffer.next_observations)::SPEC, 1, SPEC::OBSERVATION_DIM>(device, replay_buffer.next_observations, sample_index, 0);
        copy(device, device, next_observation_target, next_observation_source);

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
            typename DEVICE::index_t env_i = DETERMINISTIC ? 0 : random::uniform_int_distribution( typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, SPEC::N_ENVIRONMENTS - 1, rng);
            auto& replay_buffer = runner.replay_buffers[env_i];
            gather_batch<DEVICE, typename RUNNER::REPLAY_BUFFER_SPEC, BATCH_SPEC, RNG, DETERMINISTIC>(device, replay_buffer, batch, batch_step_i, rng);
        }
    }
//    template<typename TARGET_DEVICE, typename SOURCE_DEVICE,  typename TARGET_SPEC, typename SOURCE_SPEC>
//    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::off_policy_runner::Batch<TARGET_SPEC>& target, const nn_models::mlp::NeuralNetworkBuffersForwardBackward<SOURCE_SPEC>& source){
//        copy(target_device, source_device, (nn_models::mlp::NeuralNetworkBuffers<TARGET_SPEC>&)target, (nn_models::mlp::NeuralNetworkBuffers<SOURCE_SPEC>&)source);
//        copy(target_device, source_device, target.d_input, source.d_input);
//        copy(target_device, source_device, target.d_output, source.d_output);
//    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::off_policy_runner::Batch<TARGET_SPEC>& target, rl::components::off_policy_runner::Batch<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.observations_actions_next_observations, source.observations_actions_next_observations);
        copy(target_device, source_device, target.rewards, source.rewards);
        copy(target_device, source_device, target.terminated, source.terminated);
        copy(target_device, source_device, target.truncated, source.truncated);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::off_policy_runner::Buffers<TARGET_SPEC>& target, rl::components::off_policy_runner::Buffers<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.observations, source.observations);
        copy(target_device, source_device, target.actions, source.actions);
        copy(target_device, source_device, target.next_observations, source.next_observations);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::off_policy_runner::EpisodeStats<TARGET_SPEC>& target, rl::components::off_policy_runner::EpisodeStats<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.data, source.data);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::OffPolicyRunner<TARGET_SPEC>& target, rl::components::OffPolicyRunner<SOURCE_SPEC>& source){
        copy(target_device, source_device, target.buffers, source.buffers);
        copy(target_device, source_device, target.states, source.states);
        copy(target_device, source_device, target.episode_return, source.episode_return);
        copy(target_device, source_device, target.episode_step, source.episode_step);
        copy(target_device, source_device, target.truncated, source.truncated);
        for (typename TARGET_DEVICE::index_t env_i = 0; env_i < TARGET_SPEC::N_ENVIRONMENTS; env_i++){
            copy(target_device, source_device, target.replay_buffers[env_i], source.replay_buffers[env_i]);
            copy(target_device, source_device, target.episode_stats[env_i], source.episode_stats[env_i]);
            target.envs[env_i] = source.envs[env_i];
        }
#ifdef BACKPROP_TOOLS_DEBUG_RL_COMPONENTS_OFF_POLICY_RUNNER_CHECK_INIT
        target.initialized = source.initialized;
#endif
    }
}

#endif
