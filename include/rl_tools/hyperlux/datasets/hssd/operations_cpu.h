#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_DATASETS_HSSD_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_DATASETS_HSSD_OPERATIONS_CPU_H

#include "hssd.h"
#include "catalog.h"
#include "../glb/operations_cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::datasets::hssd {
    template <typename DEVICE>
    void enumerate(DEVICE& device, const GLB& dataset, Corpus& corpus) {
        using rl_tools::enumerate;
        enumerate(device, hyperlux::datasets::glb::Dataset{dataset.directory, dataset.references}, corpus);
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const GLB&, const Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        utils::assert_exit(device, index < corpus.references.size(), "datasets::hssd: corpus index out of range");
        using rl_tools::load;
        return load<SHADING, HAS_RGB>(device, bundle, corpus.references[index], mode);
    }
}
namespace rl_tools {
    template <typename DEVICE>
    void enumerate(DEVICE&, const hyperlux::datasets::hssd::Train125&, hyperlux::datasets::hssd::Train125::Corpus& corpus) {
        using DATASET = hyperlux::datasets::hssd::Train125;
        corpus.references.clear();
        corpus.references.reserve(DATASET::NUM_SCENES);
        for (const auto& scene : DATASET::SCENES) {
            corpus.references.emplace_back(scene.reference);
        }
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const hyperlux::datasets::hssd::Train125&, const hyperlux::datasets::hssd::Train125::Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        using LOADER = hyperlux::datasets::hssd::Train125::LOADER;
        return load<SHADING, HAS_RGB>(device, LOADER{}, corpus, index, bundle, mode);
    }

    template <typename DEVICE>
    void enumerate(DEVICE&, const hyperlux::datasets::hssd::Val43&, hyperlux::datasets::hssd::Val43::Corpus& corpus) {
        using DATASET = hyperlux::datasets::hssd::Val43;
        corpus.references.clear();
        corpus.references.reserve(DATASET::NUM_SCENES);
        for (const auto& scene : DATASET::SCENES) {
            corpus.references.emplace_back(scene.reference);
        }
    }

    template <typename SHADING = hyperlux::VeryHigh, bool HAS_RGB = true, typename DEVICE, typename T, typename MODE = mode::Default<>>
    bool load(DEVICE& device, const hyperlux::datasets::hssd::Val43&, const hyperlux::datasets::hssd::Val43::Corpus& corpus, size_t index, hyperlux::Bundle<T>& bundle, const Mode<MODE>& mode = Mode<MODE>{}) {
        using LOADER = hyperlux::datasets::hssd::Val43::LOADER;
        return load<SHADING, HAS_RGB>(device, LOADER{}, corpus, index, bundle, mode);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
