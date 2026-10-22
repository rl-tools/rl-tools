#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_HSSD_HSSD_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_HSSD_HSSD_H

#include "../glb/glb.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::hssd {
    using Corpus = glb::Corpus;

    struct GLB {
        using Corpus = hssd::Corpus;
        static constexpr const char* SEGMENTATION_TAXONOMY = "hssd-hab";
        std::string directory;
        std::vector<std::string> references;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
