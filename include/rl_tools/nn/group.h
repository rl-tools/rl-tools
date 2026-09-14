#include "../version.h"
#include "../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_NN_GROUP_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_NN_GROUP_H

#include "../utils/generic/typing.h"
#include "../numeric_types/categories.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::nn::group{
    struct Default{};

    template<typename... TAGS>
    struct Membership{
        template<typename TAG>
        static constexpr bool CONTAINS = (utils::typing::is_same_v<TAG, TAGS> || ...);
    };

    template<typename T, typename = void>
    struct MembershipOf{
        using TYPE = Membership<Default>;
    };
    template<typename T>
    struct MembershipOf<T, utils::typing::void_t<typename T::MODEL_GROUPS>>{
        using TYPE = typename T::MODEL_GROUPS;
    };

    template<typename MEMBERSHIP, typename TAG>
    struct Append;
    template<typename... TAGS, typename TAG>
    struct Append<Membership<TAGS...>, TAG>{
        using TYPE = Membership<TAGS..., TAG>;
    };
    template<typename TAG>
    struct Append<Membership<Default>, TAG>{
        using TYPE = Membership<TAG>;
    };

    template<typename... TAGS>
    struct Only{
        template<typename TARGET>
        static constexpr bool MATCH = (MembershipOf<TARGET>::TYPE::template CONTAINS<TAGS> || ...);
    };
    struct All{
        template<typename>
        static constexpr bool MATCH = true;
    };
    template<typename TAG>
    struct ParameterGroup{
        template<typename TARGET>
        static constexpr bool MATCH = utils::typing::is_same_v<typename TARGET::GROUP_TAG, TAG>;
    };
    template<typename TAG>
    struct Category{
        template<typename TARGET>
        static constexpr bool MATCH = utils::typing::is_same_v<typename TARGET::CATEGORY_TAG, TAG>;
    };
    template<typename... SELECTORS>
    struct Any{
        template<typename TARGET>
        static constexpr bool MATCH = (SELECTORS::template MATCH<TARGET> || ...);
    };
    template<typename... SELECTORS>
    struct Every{
        template<typename TARGET>
        static constexpr bool MATCH = (SELECTORS::template MATCH<TARGET> && ...);
    };
    template<typename SELECTOR>
    struct Not{
        template<typename TARGET>
        static constexpr bool MATCH = !SELECTOR::template MATCH<TARGET>;
    };

    template<typename PARAMETER_TYPE, typename GROUPS>
    struct Parameters: PARAMETER_TYPE{
        template<typename TYPE_POLICY, typename TI, typename SHAPE, typename GROUP_TAG, typename CATEGORY_TAG, bool DYNAMIC_ALLOCATION, bool CONST = false, typename NUMERIC_CATEGORY = numeric_types::categories::Parameter>
        struct Specification: PARAMETER_TYPE::template Specification<TYPE_POLICY, TI, SHAPE, GROUP_TAG, CATEGORY_TAG, DYNAMIC_ALLOCATION, CONST, NUMERIC_CATEGORY>{
            using MODEL_GROUPS = GROUPS;
        };
    };
    template<typename PARAMETER_TYPE>
    struct BaseParameters{
        using TYPE = PARAMETER_TYPE;
    };
    template<typename PARAMETER_TYPE, typename GROUPS>
    struct BaseParameters<Parameters<PARAMETER_TYPE, GROUPS>>: BaseParameters<PARAMETER_TYPE>{};
    template<typename T_CAPABILITY, typename TAG>
    struct Capability: T_CAPABILITY{
        using MODEL_GROUPS = typename Append<typename MembershipOf<T_CAPABILITY>::TYPE, TAG>::TYPE;
        using PARAMETER_TYPE = Parameters<typename BaseParameters<typename T_CAPABILITY::PARAMETER_TYPE>::TYPE, MODEL_GROUPS>;
        template<bool DYNAMIC_ALLOCATION = true, bool CONST = false>
        using CHANGE_PARAMETERS = Capability<typename T_CAPABILITY::template CHANGE_PARAMETERS<DYNAMIC_ALLOCATION, CONST>, TAG>;
    };
    template<typename T_CAPABILITY>
    struct Capability<T_CAPABILITY, Default>: T_CAPABILITY{};
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
