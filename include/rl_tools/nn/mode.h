#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODE_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn{
    namespace mode{
        struct Default{};
        struct Inference{};
    }
    template <typename MODE>
    struct Mode: MODE{};
}
#endif
