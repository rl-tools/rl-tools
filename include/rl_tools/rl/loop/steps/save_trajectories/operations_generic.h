
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_SAVE_TRAJECTORIES_OPERATIONS_GENERIC_H

#include "../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../rl/components/off_policy_runner/operations_generic.h"

#include "../../../../rl/environments/operations_generic.h"


#include "../../../../rl/utils/evaluation/operations_generic.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        malloc(device, ts.env_save_trajectories);
        ts.save_trajectories_buffer = new STATE::DATA_TYPE<typename T_CONFIG::SAVE_TRAJECTORIES_SPEC>;
        malloc(device, static_cast<typename STATE::NEXT&>(ts));
    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        init(device, ts.env_save_trajectories);
        init(device, ts.env_save_trajectories, ts.ui);
        ts.rng_save_trajectories = random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::save_trajectories::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::save_trajectories::State<T_CONFIG>;
        delete ts.save_trajectories_buffer;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::save_trajectories::State<CONFIG>& ts){
        using TS = rl::loop::steps::save_trajectories::State<CONFIG>;
        using TI = typename CONFIG::TI;
        using PARAMETERS = typename CONFIG::SAVE_TRAJECTORIES_PARAMETERS;
        using STATE = rl::loop::steps::save_trajectories::State<CONFIG>;
        if constexpr(PARAMETERS::SAVE_TRAJECTORIES == true){
            if(ts.step % PARAMETERS::INTERVAL == 0){
                evaluate(device, ts.env_eval, ts.ui, get_actor(ts), ts.save_trajectories_result, *ts.save_trajectories_buffer, ts.actor_deterministic_evaluation_buffers, ts.rng_eval, false);
            }
        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
    // to log the configuration
    template <typename DEVICE, typename PARAMETERS, typename utils::typing::enable_if<utils::typing::is_same_v<typename PARAMETERS::TAG, rl::loop::steps::save_trajectories::ParametersTag>>::type* = nullptr>
    void log(DEVICE& device, PARAMETERS){}
    template <typename DEVICE, typename CONFIG, typename utils::typing::enable_if<utils::typing::is_same_v<typename CONFIG::TAG, rl::loop::steps::save_trajectories::ConfigTag>>::type* = nullptr>
    void log(DEVICE& device, CONFIG){
        log(device, typename CONFIG::NEXT{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
