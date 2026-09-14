#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_OPTIMIZERS_UPDATE_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_OPTIMIZERS_UPDATE_H

#include "../group.h"
#include "../../mode/mode.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::optimizers{
    template<typename T_PARAMETER_SPEC>
    struct UpdateTarget{
        using PARAMETER_SPEC = T_PARAMETER_SPEC;
        using MODEL_GROUPS = typename group::MembershipOf<PARAMETER_SPEC>::TYPE;
        using GROUP_TAG = typename PARAMETER_SPEC::GROUP_TAG;
        using CATEGORY_TAG = typename PARAMETER_SPEC::CATEGORY_TAG;
    };

    namespace mode{
        template<typename T_BASE = rl_tools::mode::Default<>, typename T_SPEC = group::All>
        struct Update: T_BASE{
            using BASE = T_BASE;
            using SPEC = T_SPEC;
        };
        template<typename T_BASE = rl_tools::mode::Default<>, typename T_SPEC = group::All>
        struct Freeze: T_BASE{
            using BASE = T_BASE;
            using SPEC = T_SPEC;
        };
    }

    template<typename MODE>
    struct UpdateSelection: UpdateSelection<typename MODE::BASE>{};
    template<>
    struct UpdateSelection<rl_tools::mode::Final>{
        using TYPE = group::All;
    };
    template<typename BASE, typename SELECTOR>
    struct UpdateSelection<mode::Update<BASE, SELECTOR>>{
        using TYPE = SELECTOR;
    };
    template<typename BASE, typename SELECTOR>
    struct UpdateSelection<mode::Freeze<BASE, SELECTOR>>{
        using TYPE = group::Not<SELECTOR>;
    };

    template<typename OPTIMIZER, typename SELECTOR>
    struct Selected{
        OPTIMIZER& optimizer;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
