#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_GROUP_MODEL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_GROUP_MODEL_H

#include "../../numeric_types/categories.h"
#include "../../nn/group.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models{
    template<typename GROUP_SPEC, typename MODULE>
    struct Group{
        template<typename CAPABILITY, typename INPUT_SHAPE>
        using Layer = typename MODULE::template Layer<nn::group::Capability<CAPABILITY, GROUP_SPEC>, INPUT_SHAPE>;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
