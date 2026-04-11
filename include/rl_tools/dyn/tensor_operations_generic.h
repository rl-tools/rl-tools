#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_TENSOR_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_TENSOR_OPERATIONS_GENERIC_H

#include "model.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace dyn{
        template <typename TI>
        RL_TOOLS_FUNCTION_PLACEMENT void set_shape(Tensor<TensorSpecification<TI>>& tensor, TI rank, const TI* shape){
            tensor.rank = rank;
            for(TI i = 0; i < rank; i++) tensor.shape[i] = shape[i];
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT float get(DEVICE& device, const dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, TI flat_index){
        return dyn::to_float(reinterpret_cast<const char*>(tensor.data) + flat_index * dyn::size_of<TI>(tensor.type), tensor.type);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void set(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, float value, TI flat_index){
        dyn::from_float(reinterpret_cast<char*>(tensor.data) + flat_index * dyn::size_of<TI>(tensor.type), value, tensor.type);
    }
    template <typename DEVICE, typename TI, typename TI2, typename TI3>
    RL_TOOLS_FUNCTION_PLACEMENT float get(DEVICE& device, const dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, TI2 dim0, TI3 dim1){
        return get(device, tensor, (TI)(dim0 * tensor.shape[1] + dim1));
    }
    template <typename DEVICE, typename TI, typename TI2, typename TI3>
    RL_TOOLS_FUNCTION_PLACEMENT void set(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, float value, TI2 dim0, TI3 dim1){
        set(device, tensor, value, (TI)(dim0 * tensor.shape[1] + dim1));
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(DEVICE& d1, DEVICE& d2, const dyn::Tensor<dyn::TensorSpecification<TI>>& src, dyn::Tensor<dyn::TensorSpecification<TI>>& dst){
        for(TI i = 0; i < src.size(); i++) set(d1, dst, get(d1, src, i), i);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void scale(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor, float factor){
        for(TI i = 0; i < tensor.size(); i++) set(device, tensor, get(device, tensor, i) * factor, i);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor){
        TI s = tensor.size();
        if(s > 0){
            tensor.data = new char[s * dyn::size_of<TI>(tensor.type)];
            if(tensor.capacity == 0) tensor.capacity = s;
        }
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Tensor<dyn::TensorSpecification<TI>>& tensor){
        if(tensor.data){ delete[] reinterpret_cast<char*>(tensor.data); tensor.data = nullptr; }
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
