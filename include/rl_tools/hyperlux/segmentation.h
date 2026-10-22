#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_HYPERLUX_SEGMENTATION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_HYPERLUX_SEGMENTATION_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::hyperlux::segmentation {
    // Dataset taxonomies may not use ids from RESERVED upwards: AGENT labels the application's own
    // actors in every dataset, INHERIT keeps an overlay placement on its asset's authored classes,
    // BACKGROUND marks a miss.
    constexpr unsigned int RESERVED = 0xFFFFFF00u;
    constexpr unsigned int AGENT = 0xFFFFFFFDu;
    constexpr unsigned int INHERIT = 0xFFFFFFFEu;
    constexpr unsigned int BACKGROUND = 0xFFFFFFFFu;

    RL_TOOLS_FUNCTION_PLACEMENT constexpr bool placement_class_valid(unsigned int id){
        return id < RESERVED || id == AGENT || id == INHERIT;
    }
    RL_TOOLS_FUNCTION_PLACEMENT constexpr unsigned int effective_class(unsigned int placement, unsigned int authored){
        return placement == INHERIT ? authored : placement;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
