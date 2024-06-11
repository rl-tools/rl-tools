#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODE_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn{
    // note: please always check for the mode by using utils::typing::is_base_of_v, e.g. `utils::typing::is_base_of_v<mode::Inference, MODE>`. This ensures that when some layers of e.g. an nn_models::Sequential model are using specific modes that there are no side-effects
    namespace mode{
        struct Default{};
        struct Inference{};
    }
    template <typename MODE>
    struct Mode: MODE{};
}
#endif
