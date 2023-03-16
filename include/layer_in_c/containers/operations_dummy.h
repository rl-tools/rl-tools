#ifndef LAYER_IN_C_CONTAINERS_OPERATIONS_DUMMY_H
#define LAYER_IN_C_CONTAINERS_OPERATIONS_DUMMY_H

#include "operations_generic.h"


namespace layer_in_c{
    template<typename TARGET_DEV_SPEC, typename SOURCE_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    LAYER_IN_C_FUNCTION_PLACEMENT void copy(devices::Dummy<TARGET_DEV_SPEC>& target_device, devices::Dummy<SOURCE_DEV_SPEC>& source_device, Matrix<SPEC_1>& target, const Matrix<SPEC_2>& source){
        using TARGET_DEVICE = devices::Dummy<TARGET_DEV_SPEC>;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<TARGET_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename TARGET_DEVICE::SPEC::MATH, typename SPEC::T>>(target_device, target, source);
    }
}
#endif
