#include "../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_LOGGING_OPERATIONS_DUMMY_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_LOGGING_OPERATIONS_DUMMY_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    namespace logging{
        template <typename DEVICE, typename A>
        void text(DEVICE& device, devices::logging::Dummy* logger, const A a){
        }
        template <typename DEVICE, typename A, typename B>
        void text(DEVICE& device, devices::logging::Dummy* logger, const A a, const B b){
        }
        template <typename DEVICE, typename A, typename B, typename C, typename D>
        void text(DEVICE& device, devices::logging::Dummy* logger, const A a, const B b, const C c, const D d){
        }
    }
    template <typename DEVICE>
    void add_scalar(DEVICE& device, devices::logging::Dummy* logger, const char* key, const float value, const typename devices::logging::Dummy::index_t cadence = 1){
        //noop
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
