#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_PERSIST_H
#include <vector>
#include <cassert>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils::persist::array_conversion{
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
    void std_vector_to_vector(DEVICE& device, T source[ROWS], std::vector<T> target){
        for (typename DEVICE::index_t i=0; i < ROWS; i++){
            target[i] = source[i];
        }
    }
    template <typename DEVICE, typename SPEC>
    void std_vector_to_matrix(std::vector<std::vector<typename SPEC::T>> source, Matrix<SPEC> target){
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
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif