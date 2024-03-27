#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_TENSOR_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_TENSOR_PERSIST_H

#include <highfive/H5File.hpp>

#include "tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {

    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, HighFive::Group group, std::string dataset_name) {
        // todo
    }

    namespace tensor{
        template<typename DEVICE, typename SPEC, typename TI>
        bool check_dimensions(DEVICE& device, Tensor<SPEC>& tensor, const std::vector<TI>& dims, typename DEVICE::index_t current_dim=0){
            if constexpr(length(typename SPEC::SHAPE{}) == 0){
                return true;
            }
            else{
                auto next_tensor = view(device, tensor, current_dim);
                return dims[current_dim] == get<0>(typename SPEC::SHAPE{}) && check_dimensions(device, next_tensor, dims, current_dim+1);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, Tensor<SPEC>& tensor, const HighFive::DataSet& dataset, bool fallback_to_zero = false) {
        auto dims = dataset.getDimensions();
        static_assert(tensor::dense_layout<SPEC>(), "Load only supports dense tensors for now");
        utils::assert_exit(device, dims.size() == length(typename SPEC::SHAPE{}), "Rank mismatch");
        utils::assert_exit(device, tensor::check_dimensions(device, tensor, dims), "Dimension mismatch");
        dataset.read(data(tensor));
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, Tensor<SPEC>& tensor, HighFive::Group group, std::string dataset_name, bool fallback_to_zero = false) {
        // todo
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
