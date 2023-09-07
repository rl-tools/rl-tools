#ifndef BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_CUDA_H
#define BACKPROP_TOOLS_UTILS_ASSERT_OPERATIONS_CUDA_H
#include <cassert>
BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::CUDA<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
//            logging::text(dev.logger, message);
            assert(condition);
        }
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif