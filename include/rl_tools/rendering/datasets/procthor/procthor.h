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

    // ProcTHOR proper: the raw ai2thor-hab checkout. Scenes are converted through the
    // rl_tools_procthor_conversion library on first load and cached as GLBs (keyed by source
    // content hash + converter version, laid out as conta-addressable blobs) — targets using
    // this source must link rl_tools_procthor_conversion.
    struct AI2ThorHab {
        using Corpus = procthor::Corpus;
        std::string root;               // the directory containing configs/ (e.g. .../ai2thor-hab/ai2thor-hab)
        std::string split = "Train";    // filename filter: ProcTHOR-<split>-*.scene_instance.json
        std::string cache_directory;    // converted-GLB cache; default: ~/.cache/rl_tools/procthor_glb (dataset roots are often read-only mounts)
        bool normalize = true;          // the pre-converted corpora were produced with --normalize
    };

}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
