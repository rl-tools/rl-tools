#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_BATCH_ENVIRONMENT_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_BATCH_ENVIRONMENT_H

#include "../../../containers/tensor/tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::environments::batch {
    template <typename T_ENVIRONMENT, typename T_ENVIRONMENT::TI T_INSTANCES, bool T_DYNAMIC_ALLOCATION = true>
    struct Specification {
        using ENVIRONMENT = T_ENVIRONMENT;
        using TI = typename ENVIRONMENT::TI;
        static constexpr TI INSTANCES = T_INSTANCES;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };

    template <typename T_SPEC>
    struct Independent {
        using SPEC = T_SPEC;
        using ENVIRONMENT = typename SPEC::ENVIRONMENT;
        using T = typename ENVIRONMENT::T;
        using TI = typename SPEC::TI;
        using Parameters = typename ENVIRONMENT::Parameters;
        using State = typename ENVIRONMENT::State;
        using Observation = typename ENVIRONMENT::Observation;
        using ObservationPrivileged = typename ENVIRONMENT::ObservationPrivileged;
        static constexpr TI INSTANCES = SPEC::INSTANCES;
        static constexpr bool DYNAMIC_ALLOCATION = SPEC::DYNAMIC_ALLOCATION;
        static constexpr TI N_AGENTS = ENVIRONMENT::N_AGENTS;
        static constexpr TI ACTION_DIM = ENVIRONMENT::ACTION_DIM;
        static constexpr TI EPISODE_STEP_LIMIT = ENVIRONMENT::EPISODE_STEP_LIMIT;
        static_assert(INSTANCES > 0, "an independent batch must contain at least one environment instance");

        Tensor<tensor::Specification<ENVIRONMENT, TI, tensor::Shape<TI, INSTANCES>, SPEC::DYNAMIC_ALLOCATION>> environments;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
