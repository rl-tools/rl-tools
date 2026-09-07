#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_COLLECTION_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_COLLECTION_H
#include "on_policy_runner.h"
#include "../../../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace rl::components::on_policy_runner{
        template <typename T_SPEC>
        struct CollectionEvaluationBuffer{
            using SPEC = T_SPEC;
            using T = typename SPEC::T;
            using TI = typename SPEC::TI;
            Matrix<matrix::Specification<typename SPEC::OBSERVATION_T, TI, SPEC::N_ENVIRONMENTS, SPEC::OBSERVATION::DIM, SPEC::DYNAMIC_ALLOCATION>> observations;
            Matrix<matrix::Specification<T, TI, SPEC::N_ENVIRONMENTS, SPEC::BATCH_ENVIRONMENT::ACTION_DIM, SPEC::DYNAMIC_ALLOCATION>> actions;
        };
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
