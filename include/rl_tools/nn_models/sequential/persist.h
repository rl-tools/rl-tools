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
    template<typename DEVICE, typename SPEC, typename GROUP, typename DEVICE::index_t LAYER_I = 0>
    void save(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& model, GROUP& group) {
        using TI = typename DEVICE::index_t;
        if constexpr(LAYER_I == 0){
            set_attribute(device, group, "type", "sequential");
            write_attributes(device, group);
            group = create_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = create_group(device, group, layer_index_str);
        save(device, get_layer<LAYER_I>(model), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::NUM_LAYERS){
            save<DEVICE, SPEC, GROUP, LAYER_I+1>(device, model, group);
        }
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::sequential::ModuleForward<SPEC>& model, GROUP& group, typename DEVICE::index_t layer_i = 0) {
        using TI = typename DEVICE::index_t;
        if(layer_i == 0){
            group = get_group(device, group, "layers");
        }
        bool success = true;
        auto load_impl = [&](auto self, TI i) -> void {
            if(i < SPEC::NUM_LAYERS){
                static constexpr TI BUFFER_SIZE = 10;
                char layer_index_str[BUFFER_SIZE];
                utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, i);
                auto layer_group = get_group(device, group, layer_index_str);
                if(i == 0){ success &= load(device, get_layer<0>(model), layer_group); }
                if constexpr (SPEC::NUM_LAYERS > 1){
                    if(i == 1){ success &= load(device, get_layer<1>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 2){
                    if(i == 2){ success &= load(device, get_layer<2>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 3){
                    if(i == 3){ success &= load(device, get_layer<3>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 4){
                    if(i == 4){ success &= load(device, get_layer<4>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 5){
                    if(i == 5){ success &= load(device, get_layer<5>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 6){
                    if(i == 6){ success &= load(device, get_layer<6>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 7){
                    if(i == 7){ success &= load(device, get_layer<7>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 8){
                    if(i == 8){ success &= load(device, get_layer<8>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 9){
                    if(i == 9){ success &= load(device, get_layer<9>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 10){
                    if(i == 10){ success &= load(device, get_layer<10>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 11){
                    if(i == 11){ success &= load(device, get_layer<11>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 12){
                    if(i == 12){ success &= load(device, get_layer<12>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 13){
                    if(i == 13){ success &= load(device, get_layer<13>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 14){
                    if(i == 14){ success &= load(device, get_layer<14>(model), layer_group); }
                }
                if constexpr (SPEC::NUM_LAYERS > 15){
                    if(i == 15){ success &= load(device, get_layer<15>(model), layer_group); }
                }
                self(self, i + 1);
            }
        };
        load_impl(load_impl, layer_i);
        return success;
    }

    template<typename DEVICE, typename SPEC, typename GROUP, typename DEVICE::index_t LAYER_I = 0>
    void save(DEVICE& device, nn_models::sequential::ContentState<SPEC>& state, GROUP& group) {
        using TI = typename DEVICE::index_t;
        if constexpr(LAYER_I == 0){
            group = create_group(device, group, "layers");
        }
        static constexpr TI BUFFER_SIZE = 10;
        char layer_index_str[BUFFER_SIZE];
        utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, LAYER_I);
        auto layer_group = create_group(device, group, layer_index_str);
        save(device, get<LAYER_I>(state.states), layer_group);
        if constexpr (LAYER_I + 1 < SPEC::SPEC::NUM_LAYERS){
            save<DEVICE, SPEC, GROUP, LAYER_I+1>(device, state, group);
        }
    }

    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn_models::sequential::ContentState<SPEC>& state, GROUP& group, typename DEVICE::index_t layer_i = 0) {
        using TI = typename DEVICE::index_t;
        if(layer_i == 0){
            group = get_group(device, group, "layers");
        }
        bool success = true;
        auto load_impl = [&](auto self, TI i) -> void {
            if(i < SPEC::SPEC::NUM_LAYERS){
                static constexpr TI BUFFER_SIZE = 10;
                char layer_index_str[BUFFER_SIZE];
                utils::string::int_to_string<long int, TI>(layer_index_str, BUFFER_SIZE, i);
                auto layer_group = get_group(device, group, layer_index_str);
                if(i == 0){ success &= load(device, get<0>(state.states), layer_group); }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 1){ if(i == 1){ success &= load(device, get<1>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 2){ if(i == 2){ success &= load(device, get<2>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 3){ if(i == 3){ success &= load(device, get<3>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 4){ if(i == 4){ success &= load(device, get<4>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 5){ if(i == 5){ success &= load(device, get<5>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 6){ if(i == 6){ success &= load(device, get<6>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 7){ if(i == 7){ success &= load(device, get<7>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 8){ if(i == 8){ success &= load(device, get<8>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 9){ if(i == 9){ success &= load(device, get<9>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 10){ if(i == 10){ success &= load(device, get<10>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 11){ if(i == 11){ success &= load(device, get<11>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 12){ if(i == 12){ success &= load(device, get<12>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 13){ if(i == 13){ success &= load(device, get<13>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 14){ if(i == 14){ success &= load(device, get<14>(state.states), layer_group); } }
                if constexpr (SPEC::SPEC::NUM_LAYERS > 15){ if(i == 15){ success &= load(device, get<15>(state.states), layer_group); } }
                self(self, i + 1);
            }
        };
        load_impl(load_impl, layer_i);
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
