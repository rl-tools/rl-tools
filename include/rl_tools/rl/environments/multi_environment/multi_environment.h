#include "../../../version.h"
#include "../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_MULTI_ENVIRONMENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_MULTI_ENVIRONMENT_MULTI_ENVIRONMENT_H

#include "../../../utils/generic/typing.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments{
    namespace multi_environment{
        struct EmptySharedContext{};
        template <typename ENVIRONMENT, typename = void>
        struct SharedContext{
            using TYPE = EmptySharedContext;
        };
        template <typename ENVIRONMENT>
        struct SharedContext<ENVIRONMENT, utils::typing::void_t<typename ENVIRONMENT::SharedContext>>{
            using TYPE = typename ENVIRONMENT::SharedContext;
        };
        template <typename ENVIRONMENT, typename = void>
        struct Instances{
            static constexpr typename ENVIRONMENT::TI VALUE = 0; // 0: the member does not pin an instance count; the verbs slice by tensor shape
        };
        template <typename ENVIRONMENT>
        struct Instances<ENVIRONMENT, utils::typing::void_t<decltype(ENVIRONMENT::INSTANCES)>>{
            static constexpr typename ENVIRONMENT::TI VALUE = ENVIRONMENT::INSTANCES;
        };
    }

    // the composite is itself an environment: it satisfies the batch-verb contract by fanning
    // out to its members over contiguous instance blocks (flat, member-major). Members are
    // identical vectorized environments; shared-by-all resources live in the member-defined
    // SharedContext (empty for classic environments).
    template <typename T_ENVIRONMENT, typename T_ENVIRONMENT::TI T_NUMBER_OF_ENVIRONMENTS>
    struct MultiEnvironment{
        using ENVIRONMENT = T_ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI NUMBER_OF_ENVIRONMENTS = T_NUMBER_OF_ENVIRONMENTS;
        static_assert(NUMBER_OF_ENVIRONMENTS > 0);

        using State = typename ENVIRONMENT::State;
        using Parameters = typename ENVIRONMENT::Parameters;
        using Observation = typename ENVIRONMENT::Observation;
        using ObservationPrivileged = typename ENVIRONMENT::ObservationPrivileged;
        static constexpr TI N_AGENTS = ENVIRONMENT::N_AGENTS;
        static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
        static constexpr TI INSTANCES_PER_ENVIRONMENT = multi_environment::Instances<ENVIRONMENT>::VALUE;
        static constexpr TI INSTANCES = NUMBER_OF_ENVIRONMENTS * INSTANCES_PER_ENVIRONMENT;

        using SharedContext = typename multi_environment::SharedContext<ENVIRONMENT>::TYPE;
        SharedContext shared;
        ENVIRONMENT environments[NUMBER_OF_ENVIRONMENTS];
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
