#ifndef LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_OFF_POLICY_RUNNER_OPERATIONS_GENERIC_H

#include <layer_in_c/math/operations_generic.h>
#include "off_policy_runner.h"

#include <layer_in_c/rl/components/replay_buffer/operations_generic.h>

namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        malloc(device, buffers.observations);
        malloc(device, buffers.actions);
        malloc(device, buffers.next_observations);
    }
    template<typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        malloc(device, runner.buffers);
        malloc(device, runner.policy_eval_buffers);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            malloc(device, runner.states[env_i].replay_buffer);
        }
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI BATCH_SIZE>
    void malloc(DEVICE& device, rl::components::off_policy_runner::Batch<SPEC, BATCH_SIZE>& batch) {
        malloc(device, batch.observations);
        malloc(device, batch.actions);
        malloc(device, batch.rewards);
        malloc(device, batch.next_observations);
        malloc(device, batch.terminated);
        malloc(device, batch.truncated);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::off_policy_runner::Buffers<SPEC>& buffers) {
        free(device, buffers.observations);
        free(device, buffers.actions);
        free(device, buffers.next_observations);
    }
    template<typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner) {
        free(device, runner.buffers);
        free(device, runner.policy_eval_buffers);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++){
            free(device, runner.states[env_i].replay_buffer);
        }
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI BATCH_SIZE>
    void free(DEVICE& device, rl::components::off_policy_runner::Batch<SPEC, BATCH_SIZE>& batch) {
        free(device, batch.observations);
        free(device, batch.actions);
        free(device, batch.rewards);
        free(device, batch.next_observations);
        free(device, batch.terminated);
        free(device, batch.truncated);
    }
    template<typename DEVICE, typename SPEC, typename RNG>
    void prologue(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        // if the episode is done (step limit activated for STEP_LIMIT > 0) or if the step is the first step for this runner, reset the environment
        using RUNNER = rl::components::OffPolicyRunner<SPEC>;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        typename RUNNER::State& runner_state = runner.states[env_i];
        if (runner_state.truncated){
            sample_initial_state(device, runner_state.env, runner_state.state, rng);
            runner_state.episode_step = 0;
            runner_state.episode_return = 0;
        }
        auto observation = view<DEVICE, typename decltype(runner.buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner.buffers.observations, env_i, 0);
        observe(device, runner_state.env, runner_state.state, observation);
    }
    template<typename DEVICE, typename SPEC, typename POLICY>
    void interlude(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, POLICY &policy) {
        evaluate(device, policy, runner.buffers.observations, runner.buffers.actions, runner.policy_eval_buffers);
    }

    template<typename DEVICE, typename SPEC, typename RNG>
    void epilogue(DEVICE& device, rl::components::OffPolicyRunner<SPEC>* runner, RNG &rng, typename DEVICE::index_t env_i) {
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using PARAMETERS = typename SPEC::PARAMETERS;
        typename ENVIRONMENT::State next_state;
//        for(typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
//            auto name = "action/" + std::to_string(i);
//            add_scalar(device.logger, name.c_str(), action[i]);
//        }
        auto observation = view<DEVICE, typename decltype(runner->buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner->buffers.observations, env_i, 0);
        auto next_observation = view<DEVICE, typename decltype(runner->buffers.observations)::SPEC, 1, ENVIRONMENT::OBSERVATION_DIM>(device, runner->buffers.next_observations, env_i, 0);
        auto action = view<DEVICE, typename decltype(runner->buffers.actions)::SPEC, 1, ENVIRONMENT::ACTION_DIM>(device, runner->buffers.actions, env_i, 0);
        auto& runner_state = runner->states[env_i];

        for (typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
            T action_noisy = get(action, 0, i) + random::normal_distribution(typename DEVICE::SPEC::RANDOM(), (T) 0, PARAMETERS::EXPLORATION_NOISE, rng);
            set(action, 0, i, math::clamp<T>(action_noisy, -1, 1));
        }
//        for(typename DEVICE::index_t i = 0; i < ENVIRONMENT::ACTION_DIM; i++) {
//            auto name = "action_exploration/" + std::to_string(i);
//            add_scalar(device.logger, name.c_str(), action[i]);
//        }
        step(device, runner_state.env, runner_state.state, action, next_state);
        T reward_value = reward(device, runner_state.env, runner_state.state, action, next_state);
//        if constexpr(DEVICE::DEBUG::PRINT_REWARD){
//            std::cout << "reward: " << reward_value << std::endl;
//        }

        observe(device, runner_state.env, next_state, next_observation);

        bool terminated_flag = terminated(device, runner_state.env, next_state);
        runner_state.episode_step += 1;
        runner_state.episode_return += reward_value;
        runner_state.truncated = terminated_flag || runner_state.episode_step == SPEC::STEP_LIMIT;
        if (runner_state.truncated) {
            add_scalar(device.logger, "episode_return", runner_state.episode_return);
            add_scalar(device.logger, "episode_steps", (T)runner_state.episode_step);
        }
        // todo: add truncation / termination handling (stemming from the environment)
        add(device, runner_state.replay_buffer, observation, action, reward_value, next_observation, terminated_flag, runner_state.truncated);

        // state progression needs to come after the addition to the replay buffer because "observation" can point to the memory of runner_state.state (in the case of REQUIRES_OBSERVATION=false)
        runner_state.state = next_state;
    }

    template<typename DEVICE, typename SPEC, typename POLICY, typename RNG>
    void step(DEVICE& device, rl::components::OffPolicyRunner<SPEC> &runner, POLICY &policy, RNG &rng) {
        static_assert(POLICY::INPUT_DIM == SPEC::ENVIRONMENT::OBSERVATION_DIM,
                      "The policy's input dimension must match the environment's observation dimension.");
        static_assert(POLICY::OUTPUT_DIM == SPEC::ENVIRONMENT::ACTION_DIM,
                      "The policy's output dimension must match the environment's action dimension.");
        // todo: increase efficiency by removing the double observation of each state
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;

        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            prologue(device, runner, rng, env_i);
        }
        interlude(device, runner, policy);
        for (typename DEVICE::index_t env_i = 0; env_i < SPEC::N_ENVIRONMENTS; env_i++) {
            epilogue(device, &runner, rng, env_i);
        }
        runner.step += SPEC::N_ENVIRONMENTS;
    }

    template <typename DEVICE, typename SPEC, typename SPEC::TI BATCH_SIZE, typename RNG, bool DETERMINISTIC=false>
    void gather_batch(DEVICE& device, const rl::components::OffPolicyRunner<SPEC>& runner, rl::components::off_policy_runner::Batch<SPEC, BATCH_SIZE>& batch, RNG& rng) {
        using T = typename SPEC::T;
        for(typename DEVICE::index_t batch_step_i=0; batch_step_i < BATCH_SIZE; batch_step_i++) {
            typename DEVICE::index_t env_index = DETERMINISTIC ? 0 : random::uniform_int_distribution( typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, SPEC::N_ENVIRONMENTS - 1, rng);
            auto& replay_buffer = runner.states[env_index].replay_buffer;
            typename DEVICE::index_t sample_index_max = (replay_buffer.full ? SPEC::REPLAY_BUFFER_CAPACITY : replay_buffer.position) - 1;
            typename DEVICE::index_t sample_index = DETERMINISTIC ? batch_step_i : random::uniform_int_distribution( typename DEVICE::SPEC::RANDOM(), (typename DEVICE::index_t) 0, sample_index_max, rng);
//            utils::memcpy(&get(batch.observations, batch_step_i, 0), &get(replay_buffer.observations, sample_index, 0), SPEC::OBSERVATION_DIM);
            lic::slice(device, batch.observations, replay_buffer.observations, sample_index, 0, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM, batch_step_i, 0);
//            utils::memcpy(&get(batch.actions, batch_step_i, 0), &get(replay_buffer.actions, sample_index, 0), SPEC::ACTION_DIM);
            lic::slice(device, batch.actions, replay_buffer.actions, sample_index, 0, 1, SPEC::ENVIRONMENT::ACTION_DIM, batch_step_i, 0);
            set(batch.rewards, 0, batch_step_i, get(replay_buffer.rewards, 0, sample_index));
//            utils::memcpy(&get(batch.next_observations, batch_step_i, 0), &get(replay_buffer.next_observations, sample_index, 0), SPEC::OBSERVATION_DIM);
            lic::slice(device, batch.next_observations, replay_buffer.next_observations, sample_index, 0, 1, SPEC::ENVIRONMENT::OBSERVATION_DIM, batch_step_i, 0);
            set(batch.terminated, 0, batch_step_i, get(replay_buffer.terminated, 0, sample_index));
            set(batch.truncated, 0, batch_step_i, get(replay_buffer.truncated, 0, sample_index));
        }
    }
}

#endif
