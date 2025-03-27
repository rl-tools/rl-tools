#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_UTILS_EVALUATION_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_UTILS_EVALUATION_OPERATIONS_GENERIC_H
/*
 * This file relies on the environments methods hence it should be included after the operations of the environments that it will be used with
 */

#include "../../../math/operations_generic.h"
#include "../../environments/operations_generic.h"

#include "evaluation.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data){ }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data){ }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data){
        malloc(device, data.parameters);
        malloc(device, data.terminated);
        malloc(device, data.rewards);
        malloc(device, data.states);
        malloc(device, data.actions);
        malloc(device, data.dt);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data){
        free(device, data.parameters);
        free(device, data.terminated);
        free(device, data.rewards);
        free(device, data.states);
        free(device, data.actions);
        free(device, data.dt);
    }
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::utils::evaluation::Buffer<SPEC>& buffer){
        malloc(device, buffer.actions);
        malloc(device, buffer.observations);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::utils::evaluation::Buffer<SPEC>& buffer){
        free(device, buffer.actions);
        free(device, buffer.observations);
    }
    namespace rl::utils::evaluation{
        template<typename DEVICE, typename TI, typename SPEC>
        void set_state(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI episode_i, TI step_i, const typename SPEC::ENVIRONMENT::State& state){}
        template<typename DEVICE, typename TI, typename SPEC>
        void set_state(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI episode_i, TI step_i, const typename SPEC::ENVIRONMENT::State& state){
            set(device, data.states, state, episode_i, step_i);
        }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_parameters(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI episode_i, const typename SPEC::ENVIRONMENT::Parameters& parameters){ }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_parameters(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI episode_i, const typename SPEC::ENVIRONMENT::Parameters& parameters){
            set(device, data.parameters, parameters, episode_i);
        }
        template<typename DEVICE, typename TI, typename SPEC, typename ACTION_SPEC>
        void set_action(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI step_i, const Matrix<ACTION_SPEC>& actions){}
        template<typename DEVICE, typename TI, typename SPEC, typename ACTION_SPEC>
        void set_action(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI step_i, const Matrix<ACTION_SPEC>& actions){
            static_assert(ACTION_SPEC::ROWS == SPEC::N_EPISODES);
            static_assert(ACTION_SPEC::COLS == SPEC::ENVIRONMENT::ACTION_DIM);
            for (TI episode_i = 0; episode_i < SPEC::N_EPISODES; episode_i++){
                for (TI action_i = 0; action_i < SPEC::ENVIRONMENT::ACTION_DIM; action_i++) {
                    set(device, data.actions, get(actions, episode_i, action_i), episode_i, step_i, action_i);
                }
            }
        }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_dt(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI episode_i, TI step_i, typename SPEC::T dt){ }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_dt(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI episode_i, TI step_i, typename SPEC::T dt){
            set(device, data.dt, dt, episode_i, step_i);
        }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_reward(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI episode_i, TI step_i, typename SPEC::T reward){}
        template<typename DEVICE, typename TI, typename SPEC>
        void set_reward(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI episode_i, TI step_i, typename SPEC::T reward){
            set(device, data.rewards, reward, episode_i, step_i);
        }
        template<typename DEVICE, typename TI, typename SPEC>
        void set_terminated(DEVICE& device, rl::utils::evaluation::NoData<SPEC>& data, TI episode_i, TI step_i, bool terminated){}
        template<typename DEVICE, typename TI, typename SPEC>
        void set_terminated(DEVICE& device, rl::utils::evaluation::Data<SPEC>& data, TI episode_i, TI step_i, bool terminated){
            set(device, data.terminated, terminated, episode_i, step_i);
        }
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename POLICY_STATE, typename RNG, typename SPEC, template <typename> typename DATA, typename POLICY_EVALUATION_BUFFERS, typename MODE>
    void evaluate(DEVICE& device, ENVIRONMENT& env_init, UI& ui, const POLICY& policy, POLICY_STATE& policy_state, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, rl::utils::evaluation::Buffer<SPEC>& evaluation_buffers, rl::utils::evaluation::Result<SPEC>& results, DATA<SPEC>& data, RNG &rng, const Mode<MODE>& mode){
        using T = typename POLICY::T;
        using TI = typename DEVICE::index_t;
        constexpr TI INPUT_DIM = get_last(typename POLICY::INPUT_SHAPE{});
        constexpr TI OUTPUT_DIM = get_last(typename POLICY::OUTPUT_SHAPE{});
        static_assert(ENVIRONMENT::Observation::DIM == INPUT_DIM, "Observation and policy input dimensions must match");
        static_assert(ENVIRONMENT::ACTION_DIM == OUTPUT_DIM || (2*ENVIRONMENT::ACTION_DIM == OUTPUT_DIM), "Action and policy output dimensions must match");
        static constexpr bool STOCHASTIC_POLICY = OUTPUT_DIM == 2*ENVIRONMENT::ACTION_DIM;
        results.returns_mean = 0;
        results.returns_std = 0;
        results.episode_length_mean = 0;
        results.episode_length_std = 0;
        results.num_terminated = 0;

        auto actions_buffer = view(device, evaluation_buffers.actions, matrix::ViewSpec<SPEC::N_EPISODES, ENVIRONMENT::ACTION_DIM>{});

        ENVIRONMENT envs[SPEC::N_EPISODES];
        typename ENVIRONMENT::State states[SPEC::N_EPISODES];
        typename ENVIRONMENT::Parameters parameters[SPEC::N_EPISODES];
        bool terminated[SPEC::N_EPISODES];
        // using ADJUSTED_POLICY = typename POLICY::template CHANGE_BATCH_SIZE<TI, SPEC::N_EPISODES>;
        // typename ADJUSTED_POLICY::template State<true> policy_state;

        // malloc(device, policy_state);
        reset(device, policy, policy_state, rng);
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++){
            auto& env = envs[env_i];
            env = env_init;
            malloc(device, env);
            results.returns[env_i] = 0;
            results.episode_length[env_i] = 0;
            terminated[env_i] = false;
            auto& state = states[env_i];
            auto& current_parameters = parameters[env_i];
            if constexpr(SPEC::DETERMINISTIC_INITIAL_STATE) {
                rl_tools::initial_parameters(device, env, current_parameters);
                rl_tools::initial_state(device, env, current_parameters, state);
            }
            else{
                sample_initial_parameters(device, env, current_parameters, rng);
                sample_initial_state(device, env, current_parameters, state, rng);
            }
            rl::utils::evaluation::set_parameters(device, data, env_i, current_parameters);
        }
        for(TI step_i = 0; step_i < SPEC::STEP_LIMIT; step_i++){
            for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++){
                auto observation = row(device, evaluation_buffers.observations, env_i);
                auto& state = states[env_i];
                auto& env_parameters = parameters[env_i];
                rl::utils::evaluation::set_state(device, data, env_i, step_i, states[env_i]);
                auto& env = envs[env_i];
                observe(device, env, env_parameters, state, typename ENVIRONMENT::Observation{}, observation, rng);
            }
            auto observations_chunk = view(device, evaluation_buffers.observations, matrix::ViewSpec<SPEC::N_EPISODES, ENVIRONMENT::Observation::DIM>{}, 0, 0);
            auto actions_buffer_chunk = view(device, evaluation_buffers.actions, matrix::ViewSpec<SPEC::N_EPISODES, ENVIRONMENT::ACTION_DIM * (STOCHASTIC_POLICY ? 2 : 1)>{}, 0, 0);
            auto input_tensor = to_tensor(device, observations_chunk);
            auto output_tensor = to_tensor(device, actions_buffer_chunk);

            evaluate_step(device, policy, input_tensor, policy_state, output_tensor, policy_evaluation_buffers, rng, mode);
            if constexpr(STOCHASTIC_POLICY){ // todo: This is a special case for SAC, will be uneccessary once (https://github.com/rl-tools/rl-tools/blob/72a59eabf4038502c3be86a4f772bd72526bdcc8/TODO.md?plain=1#L22) is implemented
                for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                    for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++) {
                        set(actions_buffer, env_i, action_i, math::tanh<T>(device.math, get(actions_buffer, env_i, action_i)));
                    }
                }
            }
            rl::utils::evaluation::set_action(device, data, step_i, actions_buffer);
            for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
                if(step_i > 0){
                    if(terminated[env_i]){
                        continue;
                    }
                }
                auto& env = envs[env_i];
                typename ENVIRONMENT::State next_state;
                auto& state = states[env_i];
                auto& env_parameters = parameters[env_i];
                auto action = row(device, actions_buffer, env_i);
                T dt = step(device, env, env_parameters, state, action, next_state, rng);
                if(env_i == 0 && !terminated[env_i]){ // only render the first environment
                    set_state(device, env, env_parameters, ui, state, action);
                    render(device, env, env_parameters, ui);
                }
                rl::utils::evaluation::set_dt(device, data, env_i, step_i, dt);
                T r = reward(device, env, env_parameters, state, action, next_state, rng);
                rl::utils::evaluation::set_reward(device, data, env_i, step_i, r);
                bool terminated_flag = rl_tools::terminated(device, env, env_parameters, next_state, rng);
                if(!terminated[env_i]){
                    // count the final step as well (e.g. termination penalty)
                    results.returns[env_i] += r;
                    results.episode_length[env_i] += 1;
                    results.num_terminated += terminated_flag ? 1 : 0;
                }
                terminated_flag = terminated_flag || terminated[env_i];
                terminated[env_i] = terminated_flag;
                rl::utils::evaluation::set_terminated(device, data, env_i, step_i, terminated_flag);
                if(terminated_flag){
                    set_truncated(device, env, env_parameters, ui, next_state); // this is to sed the terminated flag to the car env
                    render(device, env, env_parameters, ui);
                }
                states[env_i] = next_state;
            }
        }
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
            auto& env_parameters = parameters[env_i];
            auto& env = envs[env_i];
            auto& state = states[env_i];
            set_truncated(device, env, env_parameters, ui, state); // this is to sed the terminated flag to the car env
            render(device, env, env_parameters, ui);
        }
        free(device, policy_state);
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++) {
            auto &env = envs[env_i];
            free(device, env);
        }
        for(TI env_i = 0; env_i < SPEC::N_EPISODES; env_i++){
            results.returns[env_i] = results.returns[env_i];
            results.returns_mean += results.returns[env_i];
            results.returns_std += results.returns[env_i]*results.returns[env_i];
            results.episode_length[env_i] = results.episode_length[env_i];
            results.episode_length_mean += results.episode_length[env_i];
            results.episode_length_std += results.episode_length[env_i]*results.episode_length[env_i];
        }
        results.returns_mean /= SPEC::N_EPISODES;
        results.returns_std = math::sqrt(device.math, math::max(device.math, (T)0, results.returns_std/SPEC::N_EPISODES - results.returns_mean*results.returns_mean));
        results.episode_length_mean /= SPEC::N_EPISODES;
        results.episode_length_std = math::sqrt(device.math, math::max(device.math, (T)0, results.episode_length_std/SPEC::N_EPISODES - results.episode_length_mean*results.episode_length_mean));
        results.share_terminated = results.num_terminated / (T)SPEC::N_EPISODES;
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename POLICY_EVALUATION_BUFFERS, typename MODE>
    void evaluate(DEVICE& device, ENVIRONMENT& env_init, UI& ui, const POLICY& policy, typename POLICY::State& policy_state, POLICY_EVALUATION_BUFFERS& policy_evaluation_buffers, rl::utils::evaluation::Buffer<SPEC>& evaluation_buffers, rl::utils::evaluation::Result<SPEC>& results, RNG &rng, const Mode<MODE>& mode){
        rl::utils::evaluation::NoData<SPEC> data;
        evaluate(device, env_init, ui, policy, policy_state, policy_evaluation_buffers, evaluation_buffers, results, data, rng, mode);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, template <typename> typename DATA, typename MODE>
    void evaluate(DEVICE& device, ENVIRONMENT& env_init, UI& ui, const POLICY& policy, rl::utils::evaluation::Result<SPEC>& results, DATA<SPEC>& data, RNG &rng, const Mode<MODE>& mode){
        using TI = typename DEVICE::index_t;
        using ADJUSTED_POLICY = typename POLICY::template CHANGE_BATCH_SIZE<TI, SPEC::N_EPISODES>;
        typename ADJUSTED_POLICY::template State<true> policy_state;
        typename ADJUSTED_POLICY::template Buffer<true> policy_evaluation_buffers;
        rl::utils::evaluation::Buffer<SPEC> evaluation_buffers;
        malloc(device, policy_state);
        malloc(device, policy_evaluation_buffers);
        malloc(device, evaluation_buffers);
        evaluate(device, env_init, ui, policy, policy_state, policy_evaluation_buffers, evaluation_buffers, results, data, rng, mode);
        free(device, policy_state);
        free(device, policy_evaluation_buffers);
        free(device, evaluation_buffers);
    }
    template<typename DEVICE, typename ENVIRONMENT, typename UI, typename POLICY, typename RNG, typename SPEC, typename MODE>
    void evaluate(DEVICE& device, ENVIRONMENT& env_init, UI& ui, const POLICY& policy, rl::utils::evaluation::Result<SPEC>& results, RNG &rng, const Mode<MODE>& mode){
        rl::utils::evaluation::NoData<SPEC> data;
        evaluate(device, env_init, ui, policy, results, data, rng, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
