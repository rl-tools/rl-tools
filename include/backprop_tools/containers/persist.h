#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_PERSIST_H

#include <highfive/H5File.hpp>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, Matrix<SPEC>& m, HighFive::Group group, std::string dataset_name) {
        using T = typename SPEC::T;
        std::vector<std::vector<T>> data(SPEC::ROWS);
        for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
            data[i] = std::vector<T>(SPEC::COLS);
            for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                data[i][j] = get(m, i, j);
            }
        }
        group.createDataSet(dataset_name, data);
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, Matrix<SPEC>& m, HighFive::Group group, std::string dataset_name, bool fallback_to_zero = false) {
        if(fallback_to_zero && !group.exist(dataset_name)){
            set_all(device, m, 0);
        }
        else{
            auto dataset = group.getDataSet(dataset_name);
            auto dims = dataset.getDimensions();
            assert(dims.size() == 2);
            assert(dims[0] == SPEC::ROWS);
            assert(dims[1] == SPEC::COLS);
            std::vector<std::vector<typename SPEC::T>> data;
            dataset.read(data);
            for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
                for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                    set(m, i, j, data[i][j]);
                }
            }
        }
    }
    template<typename DEVICE, typename SPEC, typename T>
    void load(DEVICE& device, Matrix<SPEC>& m, std::vector<std::vector<T>> data) {
        assert(data.size() == SPEC::ROWS);
        assert(data[0].size() == SPEC::COLS);
        for(typename DEVICE::index_t i=0; i < SPEC::ROWS; i++){
            for(typename DEVICE::index_t j=0; j < SPEC::COLS; j++){
                set(m, i, j, data[i][j]);
            }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
