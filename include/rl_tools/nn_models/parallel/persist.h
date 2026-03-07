#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H

#include "model.h"
#include "../sequential/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, GROUP& group) {
        set_attribute(device, group, "type", "parallel");
        write_attributes(device, group);
        auto group_a = create_group(device, group, "pipeline_a");
        save(device, model.pipeline_a, group_a);
        auto group_b = create_group(device, group, "pipeline_b");
        save(device, model.pipeline_b, group_b);
        if constexpr(SPEC::HAS_HEAD){
            auto group_head = create_group(device, group, "head");
            save(device, model.head, group_head);
        }
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, GROUP& group) {
        auto group_a = get_group(device, group, "pipeline_a");
        bool success = load(device, model.pipeline_a, group_a);
        auto group_b = get_group(device, group, "pipeline_b");
        success &= load(device, model.pipeline_b, group_b);
        if constexpr(SPEC::HAS_HEAD){
            auto group_head = get_group(device, group, "head");
            success &= load(device, model.head, group_head);
        }
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
