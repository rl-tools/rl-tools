#ifndef LAYER_IN_C_NN_LOSS_FUNCTIONS
#define LAYER_IN_C_NN_LOSS_FUNCTIONS

namespace layer_in_c::nn::loss_functions {
    template<typename DEVICE, typename T, auto DIM, auto BATCH_SIZE>
    T mse(DEVICE& device, const T a[DIM], const T b[DIM]) {
        T acc = 0;
        for(typename DEVICE::index_t i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            acc += diff * diff;
        }
        return acc / (DIM * BATCH_SIZE);
    }

    template<typename DEVICE, typename T, auto DIM, auto BATCH_SIZE>
    void d_mse_d_x(DEVICE& device, const T a[DIM], const T b[DIM], T d_a[DIM]) {
        for(typename DEVICE::index_t i = 0; i < DIM; i++) {
            T diff = a[i] - b[i];
            d_a[i] = 2*diff/(DIM * BATCH_SIZE);
        }
    }
}


#endif