#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_MODELS_BEHAVIORS_SKIP_PARAMETER_GRADIENTS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_MODELS_BEHAVIORS_SKIP_PARAMETER_GRADIENTS_H

#include "../../nn/group.h"
#include "../../mode/mode.h"
#include "../../utils/generic/tuple/tuple.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn_models::backpropagation{
    template<typename T_MODULE_SPEC, typename = void>
    struct ModuleTarget{
        using MODULE_SPEC = T_MODULE_SPEC;
        using MODEL_GROUPS = typename nn::group::MembershipOf<MODULE_SPEC>::TYPE;
    };
    template<typename T_MODULE_SPEC>
    struct ModuleTarget<T_MODULE_SPEC, utils::typing::void_t<typename T_MODULE_SPEC::CAPABILITY>>{
        using MODULE_SPEC = T_MODULE_SPEC;
        using MODEL_GROUPS = typename nn::group::MembershipOf<typename MODULE_SPEC::CAPABILITY>::TYPE;
    };

    namespace mode{
        template<typename T_BASE = rl_tools::mode::Default<>, typename T_SPEC = nn::group::All>
        struct SkipParameterGradients: T_BASE{
            using BASE = T_BASE;
            using SPEC = T_SPEC;
        };
    }

    template<typename MODE>
    struct Selection: Selection<typename MODE::BASE>{};
    template<>
    struct Selection<rl_tools::mode::Final>{
        using TYPE = nn::group::Only<>;
    };
    template<typename BASE, typename SELECTOR>
    struct Selection<mode::SkipParameterGradients<BASE, SELECTOR>>{
        using TYPE = SELECTOR;
    };

    template<typename SPEC, typename MODE>
    struct ParameterGradients{
        static constexpr bool VALUE = !Selection<MODE>::TYPE::template MATCH<ModuleTarget<SPEC>>;
    };

    template<typename MODULES, typename MODE>
    struct ParameterGradientsTuple;
    template<typename TI, typename... MODULES, typename MODE>
    struct ParameterGradientsTuple<utils::Tuple<TI, MODULES...>, MODE>{
        static constexpr bool VALUE = (ParameterGradients<typename MODULES::SPEC, MODE>::VALUE || ...);
    };
    template<typename TI, typename... MODULES, template<typename> typename MAP, typename MODE>
    struct ParameterGradientsTuple<utils::MapTuple<utils::Tuple<TI, MODULES...>, MAP>, MODE>: ParameterGradientsTuple<utils::Tuple<TI, typename MAP<MODULES>::CONTENT...>, MODE>{};

    template<typename DEVICE, typename MODULE>
    struct InputGradientOnly{
        static constexpr bool VALUE = false;
    };
    template<typename DEVICE, typename MODULE, typename MODE>
    constexpr bool INPUT_ONLY = !ParameterGradients<typename MODULE::SPEC, MODE>::VALUE && InputGradientOnly<DEVICE, MODULE>::VALUE;
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
