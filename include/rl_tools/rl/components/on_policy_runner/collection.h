#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_COLLECTION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_COLLECTION_H
#include "on_policy_runner.h"
#include "../../../mode/mode.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace mode::on_policy_runner{
        template <typename T_BASE = mode::Final, typename T_SPEC = bool>
        struct ActorOnly: T_BASE{ using BASE = T_BASE; using SPEC = T_SPEC; };
        template <typename T_BASE = mode::Final, typename T_SPEC = bool>
        struct ActorCritic: T_BASE{ using BASE = T_BASE; using SPEC = T_SPEC; };
        template <typename T_BASE = mode::Final, typename T_SPEC = bool>
        struct Sequential: T_BASE{ using BASE = T_BASE; using SPEC = T_SPEC; };
    }
    namespace rl::components::on_policy_runner{
        template <typename CRITIC, typename DATASET_SPEC, bool = DATASET_SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS>
        struct ValueState{};
        template <typename CRITIC, typename DATASET_SPEC>
        struct ValueState<CRITIC, DATASET_SPEC, true>{
            using SPEC = typename DATASET_SPEC::SPEC;
            using MODEL = typename CRITIC::template CHANGE_BATCH_SIZE<typename SPEC::TI, SPEC::N_ENVIRONMENTS>;
            typename MODEL::template State<SPEC::DYNAMIC_ALLOCATION> state;
        };
        template <typename CRITIC, typename DATASET_SPEC, bool = DATASET_SPEC::SPEC::COLLECT_NEXT_OBSERVATIONS>
        struct ValueBuffer;
        template <typename CRITIC, typename DATASET_SPEC>
        struct ValueBuffer<CRITIC, DATASET_SPEC, true>{
            using SPEC = typename DATASET_SPEC::SPEC;
            using MODEL = typename CRITIC::template CHANGE_BATCH_SIZE<typename SPEC::TI, SPEC::N_ENVIRONMENTS>;
            typename MODEL::template State<SPEC::DYNAMIC_ALLOCATION> bootstrap_state;
            typename MODEL::template Buffer<SPEC::DYNAMIC_ALLOCATION> buffer;
        };
        template <typename CRITIC, typename DATASET_SPEC>
        struct ValueBuffer<CRITIC, DATASET_SPEC, false>{
            using SPEC = typename DATASET_SPEC::SPEC;
            using MODEL = typename CRITIC::template CHANGE_BATCH_SIZE<typename SPEC::TI, DATASET_SPEC::STEPS_TOTAL_ALL>;
            typename MODEL::template Buffer<SPEC::DYNAMIC_ALLOCATION> buffer;
        };
        template <typename ACTOR, typename BUFFER>
        struct ActorEvaluation{
            ACTOR& actor;
            BUFFER& buffer;
        };
        template <typename CRITIC, typename DATASET_SPEC>
        struct ValueEvaluation{
            CRITIC& critic;
            ValueState<CRITIC, DATASET_SPEC>& state;
            ValueBuffer<CRITIC, DATASET_SPEC>& buffer;
        };
        struct NoValueEvaluation{};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
