#ifndef LAYER_IN_C_NN_LOSS_FUNCTIONS
#define LAYER_IN_C_NN_LOSS_FUNCTIONS

namespace layer_in_c::nn::loss_functions {
//    template<typename DEVICE, typename T, auto DIM, auto BATCH_SIZE>
//    T mse(DEVICE& device, const T a[DIM], const T b[DIM]) {
//        T acc = 0;
//        for(typename DEVICE::index_t i = 0; i < DIM; i++) {
//            T diff = a[i] - b[i];
//            acc += diff * diff;
//        }
//        return acc / (DIM * BATCH_SIZE);
//    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    typename SPEC_A::T mse(DEVICE& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, typename SPEC_A::T loss_weight = 1) {
        containers::check_structure<SPEC_A, SPEC_B>;
        containers::check_structure<SPEC_A, SPEC_DA>;
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        T acc = 0;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++) {
                TI index = row_i * SPEC_A::COLS + col_i;
                T diff = a[index] - b[index];
                acc += diff * diff;
            }
        }
        return acc * loss_weight / (SPEC_A::ROWS * SPEC_A::COLS);
    }

//    template<typename DEVICE, typename T, auto DIM, auto BATCH_SIZE>
//    void d_mse_d_x(DEVICE& device, const T a[DIM], const T b[DIM], T d_a[DIM], T loss_weight = 1) {
//        for(typename DEVICE::index_t i = 0; i < DIM; i++) {
//            T diff = a[i] - b[i];
//            d_a[i] = 2*diff/(DIM * BATCH_SIZE) * loss_weight;
//        }
//    }
    template<typename DEVICE, typename SPEC_A, typename SPEC_B, typename SPEC_DA>
    void d_mse_d_x(DEVICE& device, Matrix<SPEC_A> a, Matrix<SPEC_B> b, Matrix<SPEC_DA> d_a, typename SPEC_A::T loss_weight = 1) {
        containers::check_structure<SPEC_A, SPEC_B>;
        containers::check_structure<SPEC_A, SPEC_DA>;
        using T = typename SPEC_A::T;
        using TI = typename SPEC_A::TI;
        for(TI row_i = 0; row_i < SPEC_A::ROWS; row_i++) {
            for(TI col_i = 0; col_i < SPEC_A::COLS; col_i++) {
                TI index = row_i * SPEC_A::COLS + col_i;
                T diff = a.data[index] - b.data[index];
                d_a.data[index] = 2*diff/(SPEC_A::ROWS * SPEC_A::COLS) * loss_weight;
            }
        }
    }
}


#endif