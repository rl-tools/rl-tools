#include "../../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDEGUARDS) || !defined(BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_ESP32_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_ESP32_H

#include <cstdlib>
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::ESP32<DEV_SPEC>& device, bool condition, T message){
        if(!condition){
//            logging::text(device, device.logger, message);
//            throw std::runtime_error(message);
        }
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif