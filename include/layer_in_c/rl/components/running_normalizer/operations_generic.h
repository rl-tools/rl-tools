#ifndef LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_OPERATIONS_GENERIC_H
#define LAYER_IN_C_RL_COMPONENTS_RUNNING_NORMALIZER_OPERATIONS_GENERIC_H

#include "running_normalizer.h"
namespace layer_in_c{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        malloc(device, normalizer.mean);
        malloc(device, normalizer.std);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        free(device, normalizer.mean);
        free(device, normalizer.std);
    }
    template <typename DEVICE, typename SPEC>
    void init(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer){
        normalizer.age = 0;
        set_all(device, normalizer.mean, 0);
        set_all(device, normalizer.std, 1);
    }
    template <typename DEVICE, typename SPEC, typename DATA_SPEC>
    void update(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, Matrix<DATA_SPEC>& data){
        // Note: data should have >> 2 rows; subsequent calls should have the same number of rows to not skeew the mean/std
        // todo: take advantage of the data coming in batches
        static_assert(DATA_SPEC::COLS == SPEC::DIM, "Data dimension must match normalizer dimension");
        using T = typename SPEC::T;
        using TI = typename DEVICE::index_t;
        constexpr TI DATA_SIZE = DATA_SPEC::ROWS;
        static_assert(DATA_SIZE > 1, "Data size must be greater than 1 and should be much greated than one");
        normalizer.age++;
        for(TI col_i = 0; col_i < SPEC::DIM; col_i++){
            auto column = col(device, data, col_i);
            T data_mean = mean(device, column);
            T data_std = std(device, column);
            T mean = get(normalizer.mean, 0, col_i);
            T new_mean = mean + (data_mean - mean)/(normalizer.age);
            T std = get(normalizer.std, 0, col_i);
            T new_std = std + (data_std - std)/(normalizer.age);
            set(normalizer.mean, 0, col_i, new_mean);
            set(normalizer.std, 0, col_i, new_std);
        }
    }
    template <typename DEVICE, typename SPEC, typename DATA_SPEC>
    void normalize(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, Matrix<DATA_SPEC>& data){
        static_assert(DATA_SPEC::COLS == SPEC::DIM, "Data dimension must match normalizer dimension");
        normalize(device, normalizer.mean, normalizer.std, data, data);
    }
    template <typename DEVICE, typename SPEC, typename INPUT_SPEC, typename OUTPUT_SPEC>
    void normalize(DEVICE& device, rl::components::RunningNormalizer<SPEC>& normalizer, Matrix<INPUT_SPEC>& input, Matrix<OUTPUT_SPEC>& output){
        static_assert(containers::check_structure<INPUT_SPEC, OUTPUT_SPEC>);
        static_assert(INPUT_SPEC::COLS == SPEC::DIM, "Data dimension must match normalizer dimension");
        normalize(device, normalizer.mean, normalizer.std, input, output);
    }
    template <typename TARGET_DEVICE, typename SOURCE_DEVICE, typename TARGET_SPEC, typename SOURCE_SPEC>
    void copy(TARGET_DEVICE& target_device, SOURCE_DEVICE& source_device, rl::components::RunningNormalizer<TARGET_SPEC>& target, rl::components::RunningNormalizer<SOURCE_SPEC>& source){
        static_assert(TARGET_SPEC::DIM == SOURCE_SPEC::DIM, "copy: target and source normalizers must have the same dimension");
        copy(target_device, source_device, target.mean, source.mean);
        copy(target_device, source_device, target.std, source.std);
        target.age = source.age;
    }
}
#endif
