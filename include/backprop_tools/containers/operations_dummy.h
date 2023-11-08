#include "../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_CONTAINERS_OPERATIONS_DUMMY_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_DUMMY_H

#include "operations_generic.h"


BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template<typename SOURCE_DEV_SPEC, typename TARGET_DEV_SPEC, typename SPEC_1, typename SPEC_2>
    BACKPROP_TOOLS_FUNCTION_PLACEMENT void copy(devices::Dummy<SOURCE_DEV_SPEC>& source_device, devices::Dummy<TARGET_DEV_SPEC>& target_device, const Matrix<SPEC_1>& source, Matrix<SPEC_2>& target){
        using SOURCE_DEVICE = devices::Dummy<SOURCE_DEV_SPEC>;
        static_assert(containers::check_structure<SPEC_1, SPEC_2>);
        using SPEC = SPEC_1;
        vectorize_unary<SOURCE_DEVICE, SPEC_1, SPEC_2, containers::vectorization::operators::copy<typename SOURCE_DEVICE::SPEC::MATH, typename SPEC::T>>(source_device, source, target);
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
