#ifndef LAYER_IN_C_UTILS_PERSIST_H
#define LAYER_IN_C_UTILS_PERSIST_H
#include <vector>
#include <cassert>

namespace layer_in_c::utils::persist::array_conversion{
    template <typename DEVICE, typename SPEC>
    auto matrix_to_std_vector(DEVICE& device, Matrix<SPEC> M){
        using T = typename SPEC::T;
        if constexpr(SPEC::ROWS == 1){
            std::vector<T> data(SPEC::COLS);
            for(typename DEVICE::index_t i=0; i < SPEC::COLS; i++){
                data[i] = get(M, 0, i);
            }
            return data;
        }
        else{
            std::vector<std::vector<T>> data(SPEC::ROWS);
            for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
                data[i] = std::vector<T>(SPEC::COLS);
                for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                    data[i][j] = get(M, i, j);
                }
            }
            return data;
        }
    }
    template <typename DEVICE, typename T, auto ROWS>
    void std_vector_to_vector(DEVICE& device, T target[ROWS], std::vector<T> source){
        for (typename DEVICE::index_t i=0; i < ROWS; i++){
            target[i] = source[i];
        }
    }
    template <typename DEVICE, typename SPEC>
    void std_vector_to_matrix(Matrix<SPEC> target, std::vector<std::vector<typename SPEC::T>> source){
        assert(source.size() == SPEC::ROWS);
        assert(!source.empty());
        assert(source[0].size() == SPEC::COLS);
        for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
            for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                get(target, i, j) = source[i][j];
            }
        }
    }
}

#endif