#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <iostream>
#include <iomanip>
#include <cstdlib>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
#ifndef RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
    template<typename DEV_SPEC, typename SPEC>
    void malloc(devices::CPU<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data == nullptr, "Matrix is already allocated");
#endif
        // matrix._data = (typename SPEC::T*)malloc(SPEC::SIZE_BYTES);
        matrix._data = (typename SPEC::T*)std::aligned_alloc(64, SPEC::SIZE_BYTES);
        count_malloc(device, SPEC::SIZE_BYTES);
#ifdef RL_TOOLS_DEBUG_CONTAINER_MALLOC_INIT_NAN
        for(typename SPEC::TI i = 0; i < SPEC::SIZE; i++){
            if constexpr(utils::typing::is_same_v<typename SPEC::T, float> || utils::typing::is_same_v<typename SPEC::T, double>){
                matrix._data[i] = math::nan<typename SPEC::T>(typename DEV_SPEC::MATH());
            }
        }
#endif
    }
    template<typename DEV_SPEC, typename SPEC>
    void free(devices::CPU<DEV_SPEC>& device, MatrixDynamic<SPEC>& matrix){
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_MALLOC
        utils::assert_exit(device, matrix._data != nullptr, "Matrix has not been allocated");
#endif
        std::free(matrix._data);
    }
#endif
    template<typename DEVICE, typename SPEC>
    void print(DEVICE& device, const Matrix<SPEC>& m){
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                std::cout << std::fixed << std::setw(12) << std::setprecision(6) << get(m, row_i, col_i) << " ";
            }
            std::cout << std::endl;
        }
    }
    template<typename DEVICE, typename SPEC>
    void print_python_literal(DEVICE& device, const Matrix<SPEC>& m){
        std::cout << "[" << std::endl;
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            std::cout << "    [";
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                std::cout << std::fixed << std::setw(12) << std::setprecision(6) << get(m, row_i, col_i);
                if(col_i < SPEC::COLS - 1){
                    std::cout << ", ";
                }
            }
            std::cout << "],";
            std::cout << std::endl;
        }
        std::cout << "]" << std::endl;
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy_view(devices::CPU<SOURCE_DEV_SPEC>& source_device, devices::CPU<TARGET_DEV_SPEC>& target_device, const Matrix<SPEC_1>& source, Matrix<SPEC_2>& target){
        using SOURCE_DEVICE = devices::CPU<SOURCE_DEV_SPEC>;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<SOURCE_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename SOURCE_DEVICE::SPEC::MATH, typename SPEC::T>>(source_device, source, target);
    }
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(devices::CPU<SOURCE_DEV_SPEC>& source_device, devices::CPU<TARGET_DEV_SPEC>& target_device, const Matrix<SPEC_1>& source, Matrix<SPEC_2>& target){
        using SOURCE_DEVICE = devices::CPU<SOURCE_DEV_SPEC>;
        using TI = typename SOURCE_DEVICE::index_t;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        if constexpr(containers::check_memory_layout<SPEC_1, SPEC_2>){
            for(TI i = 0; i < SPEC_1::SIZE; i++){
                target._data[i] = source._data[i];
            }
        }
        else{
            copy_view(source_device, target_device, source, target);
        }
    }
    template<typename DEV_SPEC, typename SPEC, typename SPEC::TI ROWS, typename SPEC::TI COLS>
    RL_TOOLS_FUNCTION_PLACEMENT auto view(devices::CPU<DEV_SPEC>& device, const Matrix<SPEC>& m, typename SPEC::TI row, typename SPEC::TI col){
        using DEVICE = devices::CPU<DEV_SPEC>;
        static_assert(SPEC::ROWS >= ROWS);
        static_assert(SPEC::COLS >= COLS);
#ifdef RL_TOOLS_DEBUG_CONTAINER_CHECK_BOUNDS
        utils::assert_exit(device, (row + ROWS) <= SPEC::ROWS, "row + ROWS <= SPEC::ROWS");
        utils::assert_exit(device, (col + COLS) <= SPEC::COLS, "col + COLS <= SPEC::COLS");
#endif
        return _view<DEVICE, SPEC, ROWS, COLS>(device, m, row, col);
    }
    template<typename DEV_SPEC, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT std::vector<std::vector<typename SPEC::T>> std_vector(devices::CPU<DEV_SPEC>& device, Matrix<SPEC>& matrix){
        using DEVICE = devices::CPU<DEV_SPEC>;
        using TI = typename DEVICE::index_t;
        std::vector<std::vector<typename SPEC::T>> result;
        for(TI row_i = 0; row_i < SPEC::ROWS; row_i++){
            std::vector<typename SPEC::T> row;
            for(TI col_i = 0; col_i < SPEC::COLS; col_i++){
                row.push_back(get(matrix, row_i, col_i));
            }
            result.push_back(row);
        }
        return result;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
