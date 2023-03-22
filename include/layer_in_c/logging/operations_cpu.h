#ifndef LAYER_IN_C_LOGGING_OPERATIONS_CPU_H
#define LAYER_IN_C_LOGGING_OPERATIONS_CPU_H



#include <iostream>

namespace layer_in_c{
    namespace logging{
        template <typename DEVICE, typename A>
        void text(DEVICE& dev, devices::logging::CPU* logger, const A a){
            if(logger != nullptr && dev.logger == logger){
                std::cout << a << std::endl;
            }
        }
        template <typename DEVICE, typename A, typename B>
        void text(DEVICE& device, devices::logging::CPU* logger, const A a, const B b){
            if(logger != nullptr && device.logger == logger){
                std::cout << a << b << std::endl;
            }
        }
        template <typename DEVICE, typename A, typename B, typename C, typename D>
        void text(DEVICE& device, devices::logging::CPU* logger, const A a, const B b, const C c, const D d){
            if(logger != nullptr && device.logger == logger){
                std::cout << a << b << c << d << std::endl;
            }
        }
    }
    template <typename DEVICE>
    void add_scalar(DEVICE& device, devices::logging::CPU* logger, const std::string key, const float value, const typename devices::logging::CPU::index_t cadence = 1){
        //noop
    }
}
#endif
