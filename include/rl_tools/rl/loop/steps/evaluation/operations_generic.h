#include "../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_LOOP_STEPS_EVALUATION_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_LOOP_STEPS_EVALUATION_OPERATIONS_GENERIC_H

#include "../../../../rl/algorithms/sac/operations_generic.h"
#include "../../../../rl/components/off_policy_runner/operations_generic.h"

#include "../../../../rl/environments/operations_generic.h"


#include "../../../../rl/utils/evaluation.h"

#include "config.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename T_CONFIG>
    void malloc(DEVICE& device, rl::loop::steps::evaluation::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::evaluation::State<T_CONFIG>;
        malloc(device, ts.env_eval);
        malloc(device, static_cast<typename STATE::NEXT&>(ts));
    }
    template <typename DEVICE, typename T_CONFIG>
    void init(DEVICE& device, rl::loop::steps::evaluation::State<T_CONFIG>& ts, typename T_CONFIG::TI seed = 0){
        using STATE = rl::loop::steps::evaluation::State<T_CONFIG>;
        init(device, static_cast<typename STATE::NEXT&>(ts), seed);
        init(device, ts.env_eval);
        init(device, ts.env_eval, ts.ui);
        ts.rng_eval = random::default_engine(typename DEVICE::SPEC::RANDOM{}, seed);
    }

    template <typename DEVICE, typename T_CONFIG>
    void free(DEVICE& device, rl::loop::steps::evaluation::State<T_CONFIG>& ts){
        using STATE = rl::loop::steps::evaluation::State<T_CONFIG>;
        free(device, static_cast<typename STATE::NEXT&>(ts));
    }

    template <typename DEVICE, typename CONFIG>
    bool step(DEVICE& device, rl::loop::steps::evaluation::State<CONFIG>& ts){
        using TI = typename CONFIG::TI;
        using PARAMETERS = typename CONFIG::EVALUATION_PARAMETERS;
        using STATE = rl::loop::steps::evaluation::State<CONFIG>;
        if constexpr(PARAMETERS::DETERMINISTIC_EVALUATION == true){

            TI evaluation_index = ts.step / PARAMETERS::EVALUATION_INTERVAL;
            if(ts.step % PARAMETERS::EVALUATION_INTERVAL == 0 && evaluation_index < PARAMETERS::N_EVALUATIONS){
                auto result = evaluate(device, ts.env_eval, ts.ui, get_actor(ts), rl::utils::evaluation::Specification<PARAMETERS::NUM_EVALUATION_EPISODES, CONFIG::EVALUATION_PARAMETERS::EPISODE_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng_eval, false);
                log(device, device.logger, "Step: ", ts.step, "/", CONFIG::CORE_PARAMETERS::STEP_LIMIT, " Mean return: ", result.returns_mean);
                add_scalar(device, device.logger, "evaluation/return/mean", result.returns_mean);
                add_scalar(device, device.logger, "evaluation/return/std", result.returns_std);
                ts.evaluation_results[evaluation_index] = result;
            }
        }
        bool finished = step(device, static_cast<typename STATE::NEXT&>(ts));
        return finished;
    }
    template <typename DEVICE, typename PARAMETERS, typename utils::typing::enable_if<utils::typing::is_same_v<typename PARAMETERS::TAG, rl::loop::steps::evaluation::ParametersTag>>::type* = nullptr>
    void log(DEVICE& device, PARAMETERS){}
    template <typename DEVICE, typename CONFIG, typename utils::typing::enable_if<utils::typing::is_same_v<typename CONFIG::TAG, rl::loop::steps::evaluation::ConfigTag>>::type* = nullptr>
    void log(DEVICE& device, CONFIG){
        log(device, typename CONFIG::NEXT{});
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END


#endif
