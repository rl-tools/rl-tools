#ifndef LAYER_IN_C_UTILS_ASSERT_OPERATIONS_CUDA_H
#define LAYER_IN_C_UTILS_ASSERT_OPERATIONS_CUDA_H
#include <cassert>
namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::CUDA<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
//            logging::text(dev.logger, message);
            assert(condition);
        }
    }
}

#endif