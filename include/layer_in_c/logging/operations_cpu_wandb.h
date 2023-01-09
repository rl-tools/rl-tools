#ifndef LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CPU_WANDB_H
#define LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CPU_WANDB_H

#include <layer_in_c/devices/cpu.h>

#include <iostream>

namespace layer_in_c::logging{
    template <typename A>
    void text(devices::logging::CPU_WANDB& dev, const A a){

        std::cout << "Message " << dev.counter++ << ": " << a << std::endl;
    }
    template <typename A, typename B>
    void text(devices::logging::CPU_WANDB& dev, const A a, const B b){
        std::cout << "Message " << dev.counter++ << ": " << a << b << std::endl;
    }
    template <typename A, typename B, typename C, typename D>
    void text(devices::logging::CPU_WANDB& dev, const A a, const B b, const C c, const D d){
        std::cout << "Message " << dev.counter++ << ": " << a << b << c << d << std::endl;
    }
}
#endif
