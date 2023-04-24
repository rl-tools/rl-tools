#ifndef BACKPROP_TOOLS_LOGGING_OPERATIONS_CPU_H
#define BACKPROP_TOOLS_LOGGING_OPERATIONS_CPU_H



#include <iostream>

namespace backprop_tools{
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
    void set_step(DEVICE& device, devices::logging::CPU* logger, typename DEVICE::index_t step){ /* noop */ }
    template <typename DEVICE, typename ARG_1, typename ARG_2>
    void construct(DEVICE& device, devices::logging::CPU* logger, ARG_1, ARG_2){ /* noop */ }
    template <typename DEVICE>
    void construct(DEVICE& device, devices::logging::CPU* logger){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG>
    void add_scalar(DEVICE& device, devices::logging::CPU* logger, const TOPIC, const ARG){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename CADENCE>
    void add_scalar(DEVICE& device, devices::logging::CPU* logger, const TOPIC, const ARG, const CADENCE){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN, typename CADENCE>
    void add_histogram(DEVICE& device, devices::logging::CPU* logger, const TOPIC, const ARG*, const ARG_LEN, const CADENCE){ /* noop */ }
    template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN>
    void add_histogram(DEVICE& device, devices::logging::CPU* logger, const TOPIC, const ARG*, const ARG_LEN){ /* noop */ }
}
#endif
