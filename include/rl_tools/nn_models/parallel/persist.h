#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H

#include "model.h"
#include "../sequential/persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::parallel{
        template <auto I = 0, typename DEVICE, typename SPEC, typename GROUP>
        void _save_branches(DEVICE& device, ModuleForward<SPEC>& model, GROUP& group){
            if constexpr(I < SPEC::NUM_BRANCHES){
                std::string name = "branch_" + std::to_string(I);
                auto branch_group = create_group(device, group, name.c_str());
                save(device, get<I>(model.pipelines), branch_group);
                _save_branches<I + 1>(device, model, group);
            }
        }
        template <auto I = 0, typename DEVICE, typename SPEC, typename GROUP>
        bool _load_branches(DEVICE& device, ModuleForward<SPEC>& model, GROUP& group){
            if constexpr(I < SPEC::NUM_BRANCHES){
                std::string name = "branch_" + std::to_string(I);
                auto branch_group = get_group(device, group, name.c_str());
                bool success = load(device, get<I>(model.pipelines), branch_group);
                return success && _load_branches<I + 1>(device, model, group);
            }
            else{
                return true;
            }
        }
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, GROUP& group){
        set_attribute(device, group, "type", "parallel");
        using TI = typename SPEC::TI;
        constexpr TI INPUT_DIM_A = product(typename SPEC::INPUT_SHAPE_A{}) / (get<0>(typename SPEC::INPUT_SHAPE_A{}) * get<1>(typename SPEC::INPUT_SHAPE_A{}));
        constexpr TI INPUT_DIM_B = product(typename SPEC::INPUT_SHAPE_B{}) / (get<0>(typename SPEC::INPUT_SHAPE_B{}) * get<1>(typename SPEC::INPUT_SHAPE_B{}));
        set_attribute(device, group, "input_dim_a", std::to_string(INPUT_DIM_A).c_str());
        set_attribute(device, group, "input_dim_b", std::to_string(INPUT_DIM_B).c_str());
        write_attributes(device, group);
        nn_models::parallel::_save_branches(device, model, group);
        if constexpr(SPEC::HAS_HEAD){
            auto group_head = create_group(device, group, "head");
            save(device, model.head, group_head);
        }
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::parallel::ModuleForward<SPEC>& model, GROUP& group){
        bool success = nn_models::parallel::_load_branches(device, model, group);
        if constexpr(SPEC::HAS_HEAD){
            auto group_head = get_group(device, group, "head");
            success &= load(device, model.head, group_head);
        }
        return success;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
