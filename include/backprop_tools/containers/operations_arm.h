#ifndef BACKPROP_TOOLS_CONTAINERS_OPERATIONS_ARM_H
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_ARM_H

#include "operations_generic.h"
#include <cstring> // formemcpy
namespace backprop_tools{
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy_view(devices::ARM<TARGET_DEV_SPEC>& target_device, devices::ARM<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        using TARGET_DEVICE = devices::ARM<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<TARGET_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename TARGET_DEVICE::SPEC::MATH, typename SPEC::T>>(target_device, target, source);
    }
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    void copy(devices::ARM<TARGET_DEV_SPEC>& target_device, devices::ARM<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        if constexpr(containers::check_memory_layout<SPEC_1, SPEC_2>){
            std::memcpy(target._data, source._data, SPEC_1::SIZE_BYTES);
        }
        else{
            copy_view(target_device, source_device, target, source);
        }
    }
}
#endif
