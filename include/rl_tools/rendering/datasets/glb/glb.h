#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_GLB_GLB_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_GLB_GLB_H

#include <string>
#include <vector>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::glb {
    struct Corpus {
        std::vector<std::string> references;
    };

    struct Dataset {
        using Corpus = glb::Corpus;
        static constexpr const char* SEGMENTATION_TAXONOMY = "";
        std::string directory;
        std::vector<std::string> references;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
