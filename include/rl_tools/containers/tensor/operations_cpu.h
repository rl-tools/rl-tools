#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_TENSOR_OPERATIONS_CPU_H

#include "tensor.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename SPEC>
    void print(devices::CPU<DEV_SPEC>& device, Tensor<SPEC>& tensor, typename DEV_SPEC::index_t level = 0){
        using TI = typename DEV_SPEC::index_t;
        if constexpr(length(typename SPEC::SHAPE{}) == 1){
            for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                std::cout << get(device, tensor, i) << " ";
            }
            std::cout << std::endl;
        }
        else{
            if constexpr(length(typename SPEC::SHAPE{}) == 2){
                for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                    for(TI j=0; j < get<1>(typename SPEC::SHAPE{}); j++){
                        auto number = get(device, tensor, i, j);
                        std::cout <<  std::setw(10) << std::scientific << std::setprecision(3) << number;
                    }
                    std::cout << std::endl;
                }
            }
            else{
                for(TI i=0; i < get<0>(typename SPEC::SHAPE{}); i++){
                    for(TI j=0; j < level; j++){
                        std::cout << " ";
                    }
                    std::cout << "dim[" << level << "] = " << i << ": " << std::endl;
                    auto v = view(device, tensor, i);
                    print(device, v, level+1);
                }
            }
        }
    }
    template<typename DEV_SPEC, typename TI, TI VALUE, typename NEXT_ELEMENT >
    void print(devices::CPU<DEV_SPEC>& device, tensor::Element<TI, VALUE, NEXT_ELEMENT>, typename DEV_SPEC::index_t level = 0){
        using ELEMENT = tensor::Element<TI, VALUE, NEXT_ELEMENT>;
        if(level == 0){
            std::cout << "[";
        }
        if constexpr(utils::typing::is_same_v<typename NEXT_ELEMENT::NEXT_ELEMENT, tensor::FinalElement>){
            std::cout << VALUE << "]";
        }
        else{
            std::cout << VALUE << ", ";
            print(device, NEXT_ELEMENT{}, level+1);
        }
    }
}

#endif
