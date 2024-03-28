#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_TENSOR_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_TENSOR_PERSIST_H

#include <highfive/H5File.hpp>

#include "tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
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
    template<typename DEVICE, typename VECTOR, typename SPEC>
    void from_vector(DEVICE& device, const VECTOR& vector, Tensor<SPEC>& tensor) {
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            utils::assert_exit(device, vector.size() == get<0>(typename SPEC::SHAPE{}), "Vector size mismatch");
            for (TI i = 0; i < get<0>(typename SPEC::SHAPE{}); i++) {
                set(device, tensor, vector[i], i);
            }
        }
        else{
            utils::assert_exit(device, vector.size() == get<0>(typename SPEC::SHAPE{}), "Vector size mismatch");
            for (TI i = 0; i < get<0>(typename SPEC::SHAPE{}); i++) {
                auto next_tensor = view(device, tensor, i);
                from_vector(device, vector[i], next_tensor);
            }
        }
    }

    template<typename DEVICE, typename SPEC>
    auto to_vector(DEVICE& device, Tensor<SPEC>& tensor) {
        using TI = typename DEVICE::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            std::vector<typename SPEC::T> data(get<0>(typename SPEC::SHAPE{}));
            for (TI i = 0; i < get<0>(typename SPEC::SHAPE{}); i++) {
                data[i] = get(device, tensor, i);
            }
            return data;
        }
        else{
            auto next_tensor_shape = view(device, tensor, 0);
            std::vector<decltype(to_vector(device, next_tensor_shape))> result(get<0>(typename SPEC::SHAPE{}));
            for (TI i = 0; i < get<0>(typename SPEC::SHAPE{}); i++) {
                auto next_tensor = view(device, tensor, i);
                result[i] = to_vector(device, next_tensor);
            }
            return result;
        }
    }

    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, HighFive::Group group, std::string dataset_name) {
        auto data = to_vector(device, tensor);
        group.createDataSet(dataset_name, data);
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, const HighFive::DataSet& dataset, Tensor<SPEC>& tensor, bool fallback_to_zero = false) {
        using T = typename SPEC::T;
        auto dims = dataset.getDimensions();
        static_assert(tensor::dense_layout<SPEC>(), "Load only supports dense tensors for now");
        utils::assert_exit(device, dims.size() == length(typename SPEC::SHAPE{}), "Rank mismatch");
        utils::assert_exit(device, tensor::check_dimensions(device, tensor, dims), "Dimension mismatch");
        typename SPEC::T* data_ptr = data(tensor);
        utils::assert_exit(device, data_ptr != nullptr, "Data pointer is null");
        constexpr bool VIA_VECTOR = true;
        if constexpr(VIA_VECTOR){
            static_assert(!VIA_VECTOR || (length(typename SPEC::SHAPE{}) <= 3));
            if constexpr(length(typename SPEC::SHAPE{}) == 1){
                dataset.read(data_ptr);
            }
            else{
                if constexpr(length(typename SPEC::SHAPE{}) == 2){
                    std::vector<std::vector<T>> buffer;
                    dataset.read(buffer);
                    from_vector(device, buffer, tensor);
                }
                else{
                    if constexpr(length(typename SPEC::SHAPE{}) == 3){
                        std::vector<std::vector<std::vector<T>>> buffer;
                        dataset.read(buffer);
                        from_vector(device, buffer, tensor);
                    }
                }
            }
        }
        else{
            utils::assert_exit(device, dataset.getStorageSize() == sizeof(T)*SPEC::SIZE_BYTES, "Storage size mismatch");
            dataset.read(data_ptr);
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
