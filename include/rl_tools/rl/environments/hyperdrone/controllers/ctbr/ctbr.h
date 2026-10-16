#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_CTBR_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_CTBR_H

#include "../controllers.h"
#include "../../../l2f/multirotor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::hyperdrone::controllers{
    namespace ctbr{
        template<typename SPEC>
        struct State{};
        template<typename SPEC>
        struct Buffer{};
    }
    template<typename T_SPEC>
    struct CTBR{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        static constexpr TI COMMAND_DIM = 4;
        static constexpr bool SUPPORTS_SCALAR = true;
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::IDEAL_CTBR;
        static constexpr const char* NAME = "ctbr";
        using State = ctbr::State<SPEC>;
        using Buffer = ctbr::Buffer<SPEC>;
    };
    namespace ctbr{
        struct Config{
            static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::IDEAL_CTBR;
            template<typename TYPE_POLICY, typename TI, TI INSTANCES, bool DYNAMIC_ALLOCATION = true>
            using Bind = CTBR<Specification<TYPE_POLICY, TI, INSTANCES, DYNAMIC_ALLOCATION>>;
            template<typename BASE>
            struct DynamicsParameters: BASE{
                static constexpr auto ACTION_INTERFACE = Config::ACTION_INTERFACE;
                static constexpr l2f::parameters::IdealCTBR<typename BASE::PARAMETERS::T> IDEAL_CTBR{};
            };
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
