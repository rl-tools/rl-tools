#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_CACHE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_ANNOTATIONS_CACHE_H

#include <cstdlib>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::annotations {
    // opt-in persistence for derived-annotation scans (empty directory = disabled). Entries are
    // content-addressed on (artifact hash x algorithm version x scoring identity), so scenes
    // without a content hash are computed but never cached
    struct Cache {
        std::string directory;
    };

    inline std::string default_cache_directory() {
        const char* home = std::getenv("HOME");
        return (home != nullptr ? std::string(home) + "/.cache" : std::string(".")) + "/rl_tools/annotations";
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
