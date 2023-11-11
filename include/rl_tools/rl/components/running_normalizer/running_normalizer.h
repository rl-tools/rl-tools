#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components{
    namespace running_normalizer{
        template <typename T_T, typename T_TI, T_TI T_DIM, typename T_CONTAINER_TYPE_TAG = MatrixDynamicTag>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr TI DIM = T_DIM;
            using CONTAINER_TYPE_TAG = T_CONTAINER_TYPE_TAG;
        };
    }
    template <typename T_SPEC>
    struct RunningNormalizer{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::DIM;

        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, DIM>> mean;
        typename SPEC::CONTAINER_TYPE_TAG::template type<matrix::Specification<T, TI, 1, DIM>> std;
        TI age = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
