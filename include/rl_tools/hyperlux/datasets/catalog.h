#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_DATASETS_CATALOG_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_DATASETS_CATALOG_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::datasets {
    struct SceneReference {
        const char* id;
        const char* reference;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
