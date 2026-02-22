#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_SEQUENTIAL_PERSIST_H
#include "../../nn/parameters/persist.h"
#include "../../nn/persist.h"
#include "../../utils/string/operations_generic.h"
#include "model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& model, GROUP& group) {
        using TI = typename DEVICE::index_t;
        GROUP layers_group = group;
        if constexpr(LAYER_I == 0){
            set_attribute(device, group, "type", "sequential");
            write_attributes(device, group);
            layers_group = create_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = create_group(device, layers_group, layer_index_str);
        save(device, get_layer<LAYER_I>(model), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::NUM_LAYERS){
            save<LAYER_I + 1>(device, model, layers_group);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& model, GROUP& group) {
        using TI = typename DEVICE::index_t;
        GROUP layers_group = group;
        if constexpr(LAYER_I == 0){
            layers_group = get_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = get_group(device, layers_group, layer_index_str);
        bool success = load(device, get_layer<LAYER_I>(model), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::NUM_LAYERS){
            success &= load<LAYER_I + 1>(device, model, layers_group);
        }
        return success;
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::sequential::ContentState<SPEC>& state, GROUP& group) {
        using TI = typename DEVICE::index_t;
        GROUP layers_group = group;
        if constexpr(LAYER_I == 0){
            layers_group = create_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = create_group(device, layers_group, layer_index_str);
        save(device, get<LAYER_I>(state.states), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::SPEC::NUM_LAYERS){
            save<LAYER_I + 1>(device, state, layers_group);
        }
    }

    template<auto LAYER_I = 0, typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::sequential::ContentState<SPEC>& state, GROUP& group) {
        using TI = typename DEVICE::index_t;
        GROUP layers_group = group;
        if constexpr(LAYER_I == 0){
            layers_group = get_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = get_group(device, layers_group, layer_index_str);
        bool success = load(device, get<LAYER_I>(state.states), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::SPEC::NUM_LAYERS){
            success &= load<LAYER_I + 1>(device, state, layers_group);
        }
        return success;
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn_models::sequential::ModuleState<SPEC>& state, GROUP& group) {
        save(device, state.content_state, group);
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::sequential::ModuleState<SPEC>& state, GROUP& group) {
        return load(device, state.content_state, group);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
