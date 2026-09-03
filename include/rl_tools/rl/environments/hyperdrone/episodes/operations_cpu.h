#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_EPISODES_OPERATIONS_CPU_H

#include "episodes.h"
#include "operations_generic.h"
#include "../operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes){
        malloc(device, episodes.episode_step);
        malloc(device, episodes.terminated);
        malloc(device, episodes.truncated);
        malloc(device, episodes.reset);
        malloc(device, episodes.forced);
        malloc(device, episodes.episode_return);
        malloc(device, episodes.end_reason);
        malloc(device, episodes.synchronized_due);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes){
        free(device, episodes.episode_step);
        free(device, episodes.terminated);
        free(device, episodes.truncated);
        free(device, episodes.reset);
        free(device, episodes.forced);
        free(device, episodes.episode_return);
        free(device, episodes.end_reason);
        free(device, episodes.synchronized_due);
    }
    // every instance starts due for a reset: the first begin_step samples all of them
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        set_all(device, episodes.episode_step, (TI)0);
        set_all(device, episodes.terminated, false);
        set_all(device, episodes.truncated, true);
        set_all(device, episodes.reset, false);
        set_all(device, episodes.forced, false);
        set_all(device, episodes.episode_return, (T)0);
        set_all(device, episodes.end_reason, rl::environments::hyperdrone::episodes::EndReason<TI>::NONE);
        set_all(device, episodes.synchronized_due, false);
        episodes.step_limit = SPEC::STEP_LIMIT;
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const rl::environments::hyperdrone::episodes::Episodes<SOURCE_SPEC>& source, rl::environments::hyperdrone::episodes::Episodes<TARGET_SPEC>& target){
        copy(source_device, target_device, source.episode_step, target.episode_step);
        copy(source_device, target_device, source.terminated, target.terminated);
        copy(source_device, target_device, source.truncated, target.truncated);
        copy(source_device, target_device, source.reset, target.reset);
        copy(source_device, target_device, source.forced, target.forced);
        copy(source_device, target_device, source.episode_return, target.episode_return);
        copy(source_device, target_device, source.end_reason, target.end_reason);
        copy(source_device, target_device, source.synchronized_due, target.synchronized_due);
        target.step_limit = source.step_limit;
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    void malloc(DEVICE& device, rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log){
        malloc(device, log.finished_length);
        malloc(device, log.finished_return);
        malloc(device, log.finished_reason);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    void free(DEVICE& device, rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log){
        free(device, log.finished_length);
        free(device, log.finished_return);
        free(device, log.finished_reason);
    }
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS>
    void init(DEVICE& device, rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        set_all(device, log.finished_length, (T)-1);
        set_all(device, log.finished_return, (T)0);
        set_all(device, log.finished_reason, rl::environments::hyperdrone::episodes::EndReason<TI>::NONE);
    }
    template <typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename SOURCE_SPEC::TI SOURCE_STEPS, typename TARGET_SPEC, typename TARGET_SPEC::TI TARGET_STEPS>
    void copy(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const rl::environments::hyperdrone::episodes::Log<SOURCE_SPEC, SOURCE_STEPS>& source, rl::environments::hyperdrone::episodes::Log<TARGET_SPEC, TARGET_STEPS>& target){
        copy(source_device, target_device, source.finished_length, target.finished_length);
        copy(source_device, target_device, source.finished_return, target.finished_return);
        copy(source_device, target_device, source.finished_reason, target.finished_reason);
    }

    namespace rl::environments::hyperdrone::episodes {
        template <typename DEVICE, typename SPEC>
        bool _any_due(DEVICE& device, const Episodes<SPEC>& episodes){
            using TI = typename SPEC::TI;
            bool any_due = false;
            for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                any_due = any_due || _due(device, episodes, instance_i);
            }
            return any_due;
        }
        // LOG == nullptr skips the per-step log
        template <typename DEVICE, typename SPEC, typename LOG>
        void _apply_resets(DEVICE& device, Episodes<SPEC>& episodes, LOG* log, typename SPEC::TI step_i){
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            const bool any_due = SPEC::SYNCHRONIZED ? _any_due(device, episodes) : false;
            for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                const bool due = SPEC::SYNCHRONIZED ? any_due : _due(device, episodes, instance_i);
                T finished_length, finished_return;
                TI finished_reason;
                _begin_step(device, episodes, instance_i, due, finished_length, finished_return, finished_reason);
                if(log != nullptr){
                    set(device, log->finished_length, finished_length, step_i, instance_i);
                    set(device, log->finished_return, finished_return, step_i, instance_i);
                    set(device, log->finished_reason, finished_reason, step_i, instance_i);
                }
            }
        }
        template <typename ENVIRONMENT, typename SPEC>
        constexpr bool check_environment(){
            static_assert(ENVIRONMENT::INSTANCES == SPEC::INSTANCES, "the episode bookkeeping must cover all instances of the environment");
            return true;
        }
    }

    // reset decision + sampling for the due instances; call render/observe with episodes.reset afterwards
    template <typename DEVICE, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void begin_step(DEVICE& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        rl::environments::hyperdrone::episodes::_apply_resets(device, episodes, (rl::environments::hyperdrone::episodes::Log<SPEC, 1>*)nullptr, (TI)0);
        sample_initial_parameters(device, environment, parameters, episodes.reset, rng);
        sample_initial_state(device, environment, parameters, states, episodes.reset, rng);
    }
    template <typename DEVICE, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename SPEC::TI STEPS, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void begin_step(DEVICE& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log, typename SPEC::TI step_i, RNG& rng){
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        utils::assert_exit(device, step_i < STEPS, "hyperdrone::episodes::begin_step: step index outside the log");
        rl::environments::hyperdrone::episodes::_apply_resets(device, episodes, &log, step_i);
        sample_initial_parameters(device, environment, parameters, episodes.reset, rng);
        sample_initial_state(device, environment, parameters, states, episodes.reset, rng);
    }
    // after the environment step: terminal check, counters, truncation and its reason
    template <typename DEVICE, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename REWARD_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void end_step(DEVICE& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, const Tensor<REWARD_SPEC>& rewards, RNG& rng){
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            rl::environments::hyperdrone::episodes::_end_step(device, episodes, instance_i, get(device, rewards, instance_i));
        }
    }
    template <typename DEVICE, typename ENVIRONMENT, typename SPEC, typename PARAMETER_SPEC, typename STATE_SPEC, typename RNG, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void end_step(DEVICE& device, ENVIRONMENT& environment, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, Tensor<PARAMETER_SPEC>& parameters, Tensor<STATE_SPEC>& states, RNG& rng){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static_assert(rl::environments::hyperdrone::episodes::check_environment<ENVIRONMENT, SPEC>());
        terminated(device, environment, parameters, states, episodes.terminated, rng);
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            rl::environments::hyperdrone::episodes::_end_step(device, episodes, instance_i, (T)0);
        }
    }
    // pending external reset (scene rotation, epoch boundary), applied by the next begin_step
    template <typename DEVICE, typename SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void force_reset(DEVICE& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes){
        using TI = typename SPEC::TI;
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            rl::environments::hyperdrone::episodes::_force(device, episodes, instance_i);
        }
    }
    template <typename DEVICE, typename SPEC, typename MASK_SPEC, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void force_reset(DEVICE& device, rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, const Tensor<MASK_SPEC>& mask){
        using TI = typename SPEC::TI;
        static_assert(get<0>(typename MASK_SPEC::SHAPE{}) == SPEC::INSTANCES);
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            if(get(device, mask, instance_i)){
                rl::environments::hyperdrone::episodes::_force(device, episodes, instance_i);
            }
        }
    }
    // host reduction of a rollout's log (finished episodes) plus the in-progress episodes
    template <typename DEVICE, typename SPEC, typename SPEC::TI STEPS, typename utils::typing::enable_if<DEVICE::DEVICE_ID != devices::DeviceId::CUDA, bool>::type = true>
    void summarize(DEVICE& device, const rl::environments::hyperdrone::episodes::Log<SPEC, STEPS>& log, const rl::environments::hyperdrone::episodes::Episodes<SPEC>& episodes, rl::environments::hyperdrone::episodes::Statistics<typename SPEC::T, typename SPEC::TI>& statistics){
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using END_REASON = rl::environments::hyperdrone::episodes::EndReason<TI>;
        statistics = {};
        for(TI step_i = 0; step_i < STEPS; step_i++){
            for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
                const T length = get(device, log.finished_length, step_i, instance_i);
                if(length < (T)0){
                    continue;
                }
                statistics.finished++;
                statistics.length_sum += length;
                statistics.return_sum += get(device, log.finished_return, step_i, instance_i);
                const TI reason = get(device, log.finished_reason, step_i, instance_i);
                statistics.terminated += reason == END_REASON::TERMINATED ? 1 : 0;
                statistics.time_limit += reason == END_REASON::TIME_LIMIT ? 1 : 0;
                statistics.forced += reason == END_REASON::FORCED ? 1 : 0;
            }
        }
        for(TI instance_i = 0; instance_i < SPEC::INSTANCES; instance_i++){
            const TI episode_step = get(device, episodes.episode_step, instance_i);
            if(episode_step > 0){
                statistics.in_progress++;
                statistics.in_progress_length_sum += (T)episode_step;
            }
        }
        statistics.mean_length = statistics.finished > 0 ? statistics.length_sum / (T)statistics.finished : (T)0;
        statistics.mean_return = statistics.finished > 0 ? statistics.return_sum / (T)statistics.finished : (T)0;
        statistics.terminated_share = statistics.finished > 0 ? (T)statistics.terminated / (T)statistics.finished : (T)0;
        statistics.mean_in_progress_length = statistics.in_progress > 0 ? statistics.in_progress_length_sum / (T)statistics.in_progress : (T)0;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
