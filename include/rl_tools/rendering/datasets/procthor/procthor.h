#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_PROCTHOR_PROCTHOR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_PROCTHOR_PROCTHOR_H

#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::procthor {
    // pre-converted corpus: bare welded GLBs. Sources are references (paths or conta:HASH);
    // an explicit reference list wins, otherwise the directory is enumerated as a
    // lexicographically sorted .glb walk so the corpus is filesystem-order independent.
    struct Corpus {
        std::vector<std::string> references;
    };

    struct GLB {
        using Corpus = procthor::Corpus;
        std::string directory;
        std::vector<std::string> references;
    };

    template <typename T>
    struct IndoorPosition {
        T position[3];
        T yaw;
        T score;
    };

    template <typename T_T, typename T_TI, T_TI T_MAX_INDOOR_POSITIONS = 256>
    struct AnnotationsSpecification {
        using T = T_T;
        using TI = T_TI;
        static constexpr TI MAX_INDOOR_POSITIONS = T_MAX_INDOOR_POSITIONS;
    };

    // task-facing, graphics-agnostic scene annotations: free-space poses for spawning. Plain
    // trivially-copyable data — consumers mirror it into device memory for on-device sampling.
    template <typename T_SPEC>
    struct Annotations {
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;

        IndoorPosition<T> indoor_positions[SPEC::MAX_INDOOR_POSITIONS];
        TI num_indoor_positions = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
