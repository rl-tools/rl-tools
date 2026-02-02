#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EVALUATION_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EVALUATION_PERSIST_H
#include "state.h"
#include "../../../../random/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    void save(DEVICE& device, rl::loop::steps::evaluation::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::evaluation::State<T_CONFIG>;
        save(device, static_cast<typename STATE::NEXT&>(ts), group);
        auto evaluation_group = create_group(device, group, "evaluation");
        auto rng_eval_group = create_group(device, evaluation_group, "rng_eval");
        save(device, ts.rng_eval, rng_eval_group);
        auto rng_eval_on_demand_group = create_group(device, evaluation_group, "rng_eval_on_demand");
        save(device, ts.rng_eval_on_demand, rng_eval_on_demand_group);
        save_binary(device, &ts.env_eval_parameters, 1, evaluation_group, "env_eval_parameters");
    }
    template <typename DEVICE, typename T_CONFIG, typename GROUP>
    bool load(DEVICE& device, rl::loop::steps::evaluation::State<T_CONFIG>& ts, GROUP& group){
        using STATE = rl::loop::steps::evaluation::State<T_CONFIG>;
        bool success = load(device, static_cast<typename STATE::NEXT&>(ts), group);
        auto evaluation_group = get_group(device, group, "evaluation");
        auto rng_eval_group = get_group(device, evaluation_group, "rng_eval");
        bool step_result = load(device, ts.rng_eval, rng_eval_group);
        if(!step_result){ log(device, device.logger, "Evaluation loop load failed: rng_eval"); }
        success &= step_result;
        auto rng_eval_on_demand_group = get_group(device, evaluation_group, "rng_eval_on_demand");
        step_result = load(device, ts.rng_eval_on_demand, rng_eval_on_demand_group);
        if(!step_result){ log(device, device.logger, "Evaluation loop load failed: rng_eval_on_demand"); }
        success &= step_result;
        step_result = load_binary(device, &ts.env_eval_parameters, 1, evaluation_group, "env_eval_parameters");
        if(!step_result){ log(device, device.logger, "Evaluation loop load failed: env_eval_parameters"); }
        success &= step_result;
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
