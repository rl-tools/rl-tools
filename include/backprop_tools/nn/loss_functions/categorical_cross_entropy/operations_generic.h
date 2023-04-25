#ifndef BACKPROP_TOOLS_NN_LOSS_FUNCTIONS_CATEGORICAL_CROSS_ENTROPY_OPERATIONS_GENERIC
#define BACKPROP_TOOLS_NN_LOSS_FUNCTIONS_CATEGORICAL_CROSS_ENTROPY_OPERATIONS_GENERIC

namespace backprop_tools::nn::loss_functions::categorical_cross_entropy{
    template<typename DEVICE, typename SPEC_A, typename SPEC_B>
    typename SPEC_A::T evaluate(DEVICE& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, typename SPEC_A::T loss_weight = 1) {
        // a is logits, b is indices resembling empirical one-hot distributions
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        T acc = 0;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            acc += get(a, row_i, get(b, row_i, 0));
            T maximum = max(device, row(device, a, row_i));
            acc -= maximum;
            T sum = 0;
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++) {
                T logit = get(a, row_i, col_i);
                sum += math::exp(typename DEVICE::SPEC::MATH(), logit - maximum);
            }
            acc -= math::log(typename DEVICE::SPEC::MATH(), sum);
        }
        return -acc * loss_weight / SPEC_A::ROWS;
    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    void gradient(DEVICE& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, typename SPEC_A::T loss_weight = 1) {
        static_assert(containers::check_structure<SPEC_A, SPEC_B>);
        static_assert(containers::check_structure<SPEC_A, SPEC_DA>);
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            TI target_index = get(b, row_i, 0);
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++) {
                if(col_i == target_index){
                    set(d_a, row_i, col_i, -1);
                }
                T logit = get(a, row_i, col_i);

            }
        }
    }
}

#endif

