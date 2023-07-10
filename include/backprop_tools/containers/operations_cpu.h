#ifndef BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_H
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <iostream>
#include <iomanip>
namespace backprop_tools{
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
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void copy_view(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        using TARGET_DEVICE = devices::CPU<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<TARGET_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename TARGET_DEVICE::SPEC::MATH, typename SPEC::T>>(target_device, target, source);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void copy(devices::CPU<TARGET_DEV_SPEC>& target_device, devices::CPU<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        using TARGET_DEVICE = devices::CPU<TARGET_DEV_SPEC>;
        using TI = typename TARGET_DEVICE::index_t;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        if constexpr(containers::check_memory_layout<SPEC_1, SPEC_2>){
            for(TI i = 0; i < SPEC_1::SIZE; i++){
                target._data[i] = source._data[i];
            }
        }
        else{
            copy_view(target_device, source_device, target, source);
        }
    }
    template<typename DEV_SPEC, typename SPEC>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT std::vector<std::vector<typename SPEC::T>> std_vector(devices::CPU<DEV_SPEC>& device, Matrix<SPEC>& matrix){
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

#endif
