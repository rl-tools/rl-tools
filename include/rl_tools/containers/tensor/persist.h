#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_PERSIST_H

#include <highfive/H5File.hpp>

#include "tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    namespace tensor{
        template<typename DEVICE, typename SPEC, typename TI>
        bool check_dimensions(DEVICE& device, Tensor<SPEC>& tensor, const std::vector<TI>& dims, typename DEVICE::index_t current_dim=0){
            if constexpr(length(typename SPEC::SHAPE{}) == 1){
                return dims[current_dim] == get<0>(typename SPEC::SHAPE{});
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
    template<typename DEVICE, typename VT, typename SPEC>
    void from_flat_vector(DEVICE& device, const std::vector<VT>& vector, Tensor<SPEC>& tensor) {
        using T = typename SPEC::T;
        if constexpr(utils::typing::is_same_v<VT, T>){
            utils::assert_exit(device, vector.size() == SPEC::SIZE, "Vector size mismatch");
            std::memcpy(data(tensor), vector.data(), SPEC::SIZE * sizeof(T));
        }
        else{
            using TI = typename DEVICE::index_t;
            utils::assert_exit(device, vector.size() == SPEC::SIZE, "Vector size mismatch");
            std::vector<T> buffer(SPEC::SIZE);
            for (TI i = 0; i < SPEC::SIZE; i++) {
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
                buffer[i] = static_cast<VT>(vector[i]); // this is usually not called but mightt still make issues because the datatype is compile-time dispatched on the hdf5 side
#pragma GCC diagnostic pop
            }
            std::memcpy(data(tensor), buffer.data(), SPEC::SIZE * sizeof(T));
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

    namespace containers::tensor{
        template <typename SPEC, auto DIM = 0>
        void dim_helper(HighFive::DataSet& dataset){
            if constexpr(DIM < SPEC::SHAPE::LENGTH){
                std::string key = "dim_" + std::to_string(DIM);
                dataset.template createAttribute<std::string>(key, std::to_string(SPEC::SHAPE::template GET<DIM>));
                dim_helper<SPEC, DIM + 1>(dataset);
            }
        }

    }

    template<typename DEVICE, typename SPEC>
    void save(DEVICE& device, Tensor<SPEC>& tensor, HighFive::Group group, std::string dataset_name) {
        auto data = to_vector(device, tensor);
        auto dataset = group.createDataSet(dataset_name, data);
        dataset.template createAttribute<std::string>("type", "tensor");
        dataset.template createAttribute<std::string>("num_dims", std::to_string(SPEC::SHAPE::LENGTH));
        containers::tensor::dim_helper<SPEC>(dataset);
    }

    template<typename DEVICE, typename SPEC>
    void load(DEVICE& device, Tensor<SPEC>& tensor, const HighFive::Group& group, std::string dataset_name, bool fallback_to_zero = false) {
        using T = typename SPEC::T;
        if(fallback_to_zero && !group.exist(dataset_name)){
            set_all(device, tensor, 0);
        }
        else{
            auto dataset = group.getDataSet(dataset_name);
            auto dims = dataset.getDimensions();
            static_assert(tensor::dense_row_major_layout<SPEC>(), "Load only supports dense tensors for now");
            utils::assert_exit(device, dims.size() == length(typename SPEC::SHAPE{}), "Rank mismatch");
            utils::assert_exit(device, tensor::check_dimensions(device, tensor, dims), "Dimension mismatch");
            typename SPEC::T* data_ptr = data(tensor);
            utils::assert_exit(device, data_ptr != nullptr, "Data pointer is null");
            auto data_type = dataset.getDataType();
            auto data_type_class = data_type.getClass();
            auto data_type_size = data_type.getSize();
            utils::assert_exit(device, data_type_class == HighFive::DataTypeClass::Float || data_type_class == HighFive::DataTypeClass::Integer, "Only Float and Int are currently supported");
            utils::assert_exit(device, data_type_size == 4 || data_type_size == 8, "Only Float32 and Float64 are currently supported");
            utils::assert_exit(device, dataset.getStorageSize() == data_type_size * SPEC::SIZE, "Storage size mismatch");
            if (data_type_class == HighFive::DataTypeClass::Float){
                if(data_type_size == 4){
                    std::vector<float> buffer(SPEC::SIZE);
                    dataset.read(buffer.data());
                    from_flat_vector(device, buffer, tensor);
                }
                else{
                    if(data_type_size == 8){
                        std::vector<double> buffer(SPEC::SIZE);
                        dataset.read(buffer.data());
                        from_flat_vector(device, buffer, tensor);
                    }
                    else{
                        utils::assert_exit(device, false, "Unsupported data type size");
                    }
                }
            }
            else{
                if(data_type_class == HighFive::DataTypeClass::Integer){
                    if(data_type_size == 4){
                        std::vector<int32_t> buffer(SPEC::SIZE);
                        dataset.read(buffer.data());
                        from_flat_vector(device, buffer, tensor);
                    }
                    else{
                        if(data_type_size == 8){
                            std::vector<int64_t> buffer(SPEC::SIZE);
                            dataset.read(buffer.data());
                            from_flat_vector(device, buffer, tensor);
                        }
                        else{
                            utils::assert_exit(device, false, "Unsupported data type size");
                        }
                    }
                }
            }
            // if(data_type_size == 4){
            //     std::vector<float> buffer(SPEC::SIZE);
            //     dataset.read(buffer.data()); // we use the .data() pointer here because otherwise HighFive will complain about a multi-dimensional dataset. In this way we can load (the assumedly dense data directly)
            //     from_flat_vector(device, buffer, tensor);
            // }
            // else{
            //     if(data_type_size == 8){
            //         std::vector<double> buffer(SPEC::SIZE);
            //         dataset.read(buffer.data());
            //         from_flat_vector(device, buffer, tensor);
            //     }
            //     else{
            //         utils::assert_exit(device, false, "Unsupported data type size");
            //     }
            // }
        }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
