#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_DATASETS_WRAPPERS_FREE_SPACE_BOX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_DATASETS_WRAPPERS_FREE_SPACE_BOX_H

#include <limits>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rendering::datasets::wrappers {
    // dataset wrapper restricting the free-space table to an axis-aligned box — the prototypical
    // task-side customization: wraps any dataset (or wrapper), forwards enumerate/load, and
    // subsets the annotation product. The default box is unbounded (no-op)
    template <typename T_INNER, typename T_T>
    struct FreeSpaceBox {
        using INNER = T_INNER;
        using T = T_T;
        using Corpus = typename INNER::Corpus;
        INNER inner;
        T min[3] = {std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest(), std::numeric_limits<T>::lowest()};
        T max[3] = {std::numeric_limits<T>::max(), std::numeric_limits<T>::max(), std::numeric_limits<T>::max()};
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
