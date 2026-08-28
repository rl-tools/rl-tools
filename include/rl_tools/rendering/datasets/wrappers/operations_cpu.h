#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_WRAPPERS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_WRAPPERS_OPERATIONS_CPU_H

#include "free_space_box.h"
#include "../glb/operations_cpu.h"
#include "../annotations/operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::wrappers {
    template <typename DEVICE, typename INNER, typename WRAPPER_T>
    void enumerate(DEVICE& device, const FreeSpaceBox<INNER, WRAPPER_T>& dataset, typename FreeSpaceBox<INNER, WRAPPER_T>::Corpus& corpus) {
        enumerate(device, dataset.inner, corpus);
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename INNER, typename WRAPPER_T, typename T>
    bool load(DEVICE& device, const FreeSpaceBox<INNER, WRAPPER_T>& dataset, const typename FreeSpaceBox<INNER, WRAPPER_T>::Corpus& corpus, size_t index, rendering::Bundle<T>& bundle) {
        return load<SHADING, HAS_RGB>(device, dataset.inner, corpus, index, bundle);
    }

    // the annotation table stays sorted: the filter preserves the inner producer's score order
    template <typename DEVICE, typename INNER, typename WRAPPER_T, typename INDEX_TI, typename SPEC, typename METADATA_T, typename PROBE, typename PARAMETERS_T, typename PARAMETERS_TI>
    void annotate(DEVICE& device, const FreeSpaceBox<INNER, WRAPPER_T>& dataset, const typename FreeSpaceBox<INNER, WRAPPER_T>::Corpus& corpus, INDEX_TI index, annotations::FreeSpace<SPEC>& product, const rendering::SceneMetadata<METADATA_T>& metadata, PROBE& probe_target, const annotations::FreeSpaceParameters<PARAMETERS_T, PARAMETERS_TI>& parameters, const annotations::Cache& cache) {
        using TI = typename SPEC::TI;
        annotate(device, dataset.inner, corpus, index, product, metadata, probe_target, parameters, cache);
        TI kept = 0;
        for (TI position_i = 0; position_i < product.num_positions; position_i++) {
            const auto& position = product.positions[position_i];
            bool inside = true;
            for (unsigned axis = 0; axis < 3; axis++) {
                inside = inside && position.position[axis] >= dataset.min[axis] && position.position[axis] <= dataset.max[axis];
            }
            if (inside) {
                product.positions[kept] = position;
                kept++;
            }
        }
        product.num_positions = kept;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
