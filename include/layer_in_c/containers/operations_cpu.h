#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_CPU_H

#include "operations_generic.h"

#include <iostream>
namespace layer_in_c{
    template<typename DEVICE, typename SPEC>
    void print(DEVICE& device, Matrix<SPEC>& m){
        for(typename DEVICE::index_t row_i = 0; row_i < SPEC::ROWS; row_i++){
            for(typename DEVICE::index_t col_i = 0; col_i < SPEC::COLS; col_i++){
                std::cout << get(m, row_i, col_i) << " ";
            }
            std::cout << std::endl;
        }
    }
}

#endif
