#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::rl::components{
    namespace running_normalizer{
        template <typename T_TYPE_POLICY, typename T_TI, T_TI T_DIM>
        struct Specification{
            using TYPE_POLICY = T_TYPE_POLICY;
            using TI = T_TI;
            static constexpr TI DIM = T_DIM;
        };
    }
    template <typename T_SPEC>
    struct RunningNormalizer{
        using SPEC = T_SPEC;
        using TYPE_POLICY = typename SPEC::TYPE_POLICY;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::DIM;

        using T = typename TYPE_POLICY::DEFAULT;
        Matrix<matrix::Specification<T, TI, 1, DIM>> mean;
        Matrix<matrix::Specification<T, TI, 1, DIM>> std;
        TI age = 0;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
