#ifndef LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_OPERATIONS_GENERIC_H

#include "running_normalizer.h"
namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        malloc(device, normalizer.mean);
        malloc(device, normalizer.variance);
        malloc(device, normalizer.std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        free(device, normalizer.mean);
        free(device, normalizer.variance);
        free(device, normalizer.std);
    }
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        normalizer.age = 0;
        set_all(device, normalizer.mean, 0);
        set_all(device, normalizer.variance, 0);
        set_all(device, normalizer.std, 0);
    }
    template <typename DEVICE, typename SPEC, typename DATA_SPEC>
    void update(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, Matrix<DATA_SPEC>& data){
        static_assert(DATA_SPEC::COLS == SPEC::DIM, "Data dimension must match normalizer dimension");
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI DATA_SIZE = DATA_SPEC::ROWS;
        for(TI row_i = 0; row_i < DATA_SIZE; row_i++){
            normalizer.age++;
            for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
                T x = get(data, row_i, col_i);
                T mean = get(normalizer.mean, 0, col_i);
                T variance = get(normalizer.variance, 0, col_i);
                T new_mean = mean + (x - mean)/normalizer.age;
                T new_variance = variance + x * x / normalizer.age;
                set(normalizer.mean, 0, col_i, new_mean);
                set(normalizer.variance, 0, col_i, new_variance);
            }
        }
        for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
            T mean = get(normalizer.mean, 0, col_i);
            T variance = get(normalizer.variance, 0, col_i);
            variance -= mean * mean;
            T std = math::sqrt(typename DEVICE::SPEC::MATH(), variance);
            set(normalizer.std, 0, col_i, std);
        }
    }
    template <typename DEVICE, typename SPEC, typename DATA_SPEC>
    void normalize(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, Matrix<DATA_SPEC>& data){
        static_assert(DATA_SPEC::COLS == SPEC::DIM, "Data dimension must match normalizer dimension");
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI DATA_SIZE = DATA_SPEC::ROWS;
        for(TI row_i = 0; row_i < DATA_SIZE; row_i++){
            for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
                T x = get(data, row_i, col_i);
                T mean = get(normalizer.mean, 0, col_i);
                T std = get(normalizer.std, 0, col_i);
                T normalized_x = (x - mean) / std;
                set(data, row_i, col_i, normalized_x);
            }
        }
    }
}
#endif
