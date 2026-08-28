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

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
