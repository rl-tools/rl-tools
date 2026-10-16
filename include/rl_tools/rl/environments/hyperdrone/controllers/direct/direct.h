#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_H

#include "../controllers.h"
#include "../../../l2f/multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::controllers{
    namespace direct{
        template<typename SPEC>
        struct State{};
        template<typename SPEC>
        struct Buffer{};
    }
    template<typename T_SPEC, l2f::parameters::ActionInterface T_ACTION_INTERFACE>
    struct Direct{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        static constexpr TI COMMAND_DIM = 4;
        static constexpr bool SUPPORTS_SCALAR = true;
        static constexpr auto ACTION_INTERFACE = T_ACTION_INTERFACE;
        static constexpr const char* NAME = T_ACTION_INTERFACE == l2f::parameters::ActionInterface::DIRECT_MOTOR ? "motor" : "ctbr_pd";
        using State = direct::State<SPEC>;
        using Buffer = direct::Buffer<SPEC>;
    };
    namespace direct{
        template<l2f::parameters::ActionInterface T_ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR>
        struct Config{
            static_assert(T_ACTION_INTERFACE == l2f::parameters::ActionInterface::DIRECT_MOTOR || T_ACTION_INTERFACE == l2f::parameters::ActionInterface::CTBR);
            static constexpr auto ACTION_INTERFACE = T_ACTION_INTERFACE;
            template<typename TYPE_POLICY, typename TI, TI INSTANCES, bool DYNAMIC_ALLOCATION = true>
            using Bind = Direct<Specification<TYPE_POLICY, TI, INSTANCES, DYNAMIC_ALLOCATION>, ACTION_INTERFACE>;
            template<typename BASE>
            struct DynamicsParameters: BASE{
                static constexpr auto ACTION_INTERFACE = Config::ACTION_INTERFACE;
            };
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
