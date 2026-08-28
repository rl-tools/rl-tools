#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_PROCTHOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_PROCTHOR_OPERATIONS_CPU_H

#include "procthor.h"
#include "../operations_cpu.h"
#include "../glb/operations_cpu.h"

#include <algorithm>
#include <filesystem>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::procthor {
    template <typename DEVICE>
    void enumerate(DEVICE& device, const GLB& dataset, Corpus& corpus) {
        corpus.references.clear();
        if(!dataset.references.empty()){
            corpus.references = dataset.references;
            return;
        }
        for (const auto& entry : std::filesystem::directory_iterator(dataset.directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".glb") {
                corpus.references.push_back(entry.path().string());
            }
        }
        std::sort(corpus.references.begin(), corpus.references.end());
        utils::assert_exit(device, !corpus.references.empty(), "datasets::procthor::GLB: no .glb scenes found in directory");
    }

    template <typename SHADING = rendering::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T>
    bool load(DEVICE& device, const GLB& dataset, const Corpus& corpus, size_t index, rendering::Bundle<T>& bundle) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::procthor: corpus index out of range");
        return rl_tools::load<SHADING, HAS_RGB>(device, bundle, corpus.references[index]);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
