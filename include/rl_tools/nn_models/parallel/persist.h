#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_PARALLEL_PERSIST_H

#include "model.h"
#include "../sequential/persist.h"

#include <sstream>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace nn_models::parallel{
        template <typename SHAPE, auto INDEX = 0>
        void _shape_to_csv(std::stringstream& ss){
            if constexpr(INDEX == 0){
                ss << get<0>(SHAPE{});
            }
            else{
                ss << "," << get<INDEX>(SHAPE{});
            }
            if constexpr(INDEX + 1 < length(SHAPE{})){
                _shape_to_csv<SHAPE, INDEX + 1>(ss);
            }
        }

        template <auto I = 0, typename DEVICE, typename SPEC, typename GROUP>
        void _save_branches(DEVICE& device, ModuleForward<SPEC>& model, GROUP& group){
            if constexpr(I < SPEC::NUM_BRANCHES){
                std::string name = "branch_" + std::to_string(I);
                auto branch_group = create_group(device, group, name.c_str());
                using BRANCH = typename utils::tuple_element<I, typename SPEC::BRANCH_TUPLE>::type;
                using INPUT_SHAPE = typename BRANCH::INPUT_SHAPE;
                std::stringstream shape_ss;
                _shape_to_csv<INPUT_SHAPE>(shape_ss);
                set_attribute(device, branch_group, "input_shape", shape_ss.str().c_str());
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
        set_attribute(device, group, "num_branches", std::to_string(SPEC::NUM_BRANCHES).c_str());
        set_attribute(device, group, "has_head", SPEC::HAS_HEAD ? "1" : "0");
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
