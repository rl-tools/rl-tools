#ifndef LAYER_IN_C_UTILS_ASSERT_OPERATIONS_ESP32_H
#define LAYER_IN_C_UTILS_ASSERT_OPERATIONS_ESP32_H

#include <cstdlib>
namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::ESP32<DEV_SPEC>& device, bool condition, T message){
        if(!condition){
//            logging::text(device, device.logger, message);
//            throw std::runtime_error(message);
        }
    }
}

#endif