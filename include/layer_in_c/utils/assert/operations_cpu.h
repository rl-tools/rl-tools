#ifndef LAYER_IN_C_ASSERT_OPERATIONS_CPU_H
#define LAYER_IN_C_ASSERT_OPERATIONS_CPU_H

#include <cstdlib>
namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert(devices::CPU<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            logging::text(dev.logger, message);
            std::exit(-1);
        }
    }
}

#endif