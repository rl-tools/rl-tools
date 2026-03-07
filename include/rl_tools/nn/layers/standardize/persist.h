#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_LAYERS_STANDARDIZE_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_LAYERS_STANDARDIZE_PERSIST_H
#include "layer.h"
#include "../../parameters/persist.h"
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer, GROUP& group) {
        auto mean_group = create_group(device, group, "mean");
        save(device, layer.mean, mean_group);
        auto precision_group = create_group(device, group, "precision");
        save(device, layer.precision, precision_group);
        auto running_mean_group = create_group(device, group, "running_mean");
        save(device, layer.running_mean, running_mean_group);
        write_attributes(device, running_mean_group);
        auto running_std_group = create_group(device, group, "running_std");
        save(device, layer.running_std, running_std_group);
        write_attributes(device, running_std_group);
        {
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            auto age_group = create_group(device, group, "age");
            Matrix<matrix::Specification<T, typename SPEC::TI, 1, 1, false>> age_matrix;
            set(age_matrix, 0, 0, layer.age);
            save(device, age_matrix, age_group, "value");
            write_attributes(device, age_group);
        }
        set_attribute(device, group, "type", "standardize");
        write_attributes(device, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::standardize::LayerBackward<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::standardize::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    void save(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, GROUP& group) {
        save(device, (nn::layers::standardize::LayerBackward<SPEC>&)layer, group);
        save(device, layer.output, group, "output");
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::standardize::LayerForward<SPEC>& layer, GROUP& group) {
        auto mean_group = get_group(device, group, "mean");
        bool success = load(device, layer.mean, mean_group);
        auto precision_group = get_group(device, group, "precision");
        success &= load(device, layer.precision, precision_group);
        if(group_exists(device, group, "running_mean")){
            auto running_mean_group = get_group(device, group, "running_mean");
            success &= load(device, layer.running_mean, running_mean_group);
        }
        if(group_exists(device, group, "running_std")){
            auto running_std_group = get_group(device, group, "running_std");
            success &= load(device, layer.running_std, running_std_group);
        }
        if(group_exists(device, group, "age")){
            using T = typename SPEC::TYPE_POLICY::DEFAULT;
            auto age_group = get_group(device, group, "age");
            Matrix<matrix::Specification<T, typename SPEC::TI, 1, 1, false>> age_matrix;
            set(age_matrix, 0, 0, (T)0);
            load(device, age_matrix, age_group, "value");
            layer.age = get(age_matrix, 0, 0);
        }
        return success;
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::standardize::LayerBackward<SPEC>& layer, GROUP& group) {
        return load(device, (nn::layers::standardize::LayerForward<SPEC>&)layer, group);
    }
    template<typename DEVICE, typename SPEC, typename GROUP>
    bool load(DEVICE& device, nn::layers::standardize::LayerGradient<SPEC>& layer, GROUP& group) {
        bool success = load(device, (nn::layers::standardize::LayerBackward<SPEC>&)layer, group);
        if(group_exists(device, group, "output")){
            success &=  load(device, layer.output, group, "output");
        }
        return success;
    }
    template<typename DEVICE, typename GROUP>
    void save(DEVICE& device, nn::layers::standardize::State& state, GROUP& group) {}
    template<typename DEVICE, typename GROUP>
    bool load(DEVICE& device, nn::layers::standardize::State& state, GROUP& group) { return true; }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
