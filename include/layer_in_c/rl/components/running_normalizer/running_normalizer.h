#ifndef LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H
#define LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_RUNNING_NORMALIZER_H

namespace layer_in_c::rl::components{
    namespace running_normalizer{
        template <typename T_T, typename T_TI, T_TI T_DIM>
        struct Specification{
            using T = T_T;
            using TI = T_TI;
            static constexpr TI DIM = T_DIM;
        };
    }
    template <typename T_SPEC>
    struct RunningNormalizer{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        static constexpr TI DIM = SPEC::DIM;

        Matrix<matrix::Specification<T, TI, 1, DIM>> mean;
        Matrix<matrix::Specification<T, TI, 1, DIM>> variance;
        Matrix<matrix::Specification<T, TI, 1, DIM>> std;
        TI age = 0;
    };
}

#endif
