#ifndef LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CUDA_H
#define LAYER_IN_C_UTILS_LOGGING_OPERATIONS_CUDA_H


#include <layer_in_c/devices/cuda.h>

namespace layer_in_c{
    namespace logging{
        template <typename A>
        void text(devices::logging::CUDA& dev, const char * a, const char * b){
            std::cout << a << b << std::endl;
        }
        template <typename A>
        void text(devices::logging::CUDA& dev, const A a){
        }
        template <typename A, typename B>
        void text(devices::logging::CUDA& dev, const A a, const B b){
        }
        template <typename A, typename B, typename C, typename D>
        void text(devices::logging::CUDA& dev, const A a, const B b, const C c, const D d){
        }
        void add_scalar(devices::logging::CUDA& dev, const char* key, const float value, const typename devices::logging::CUDA::index_t cadence = 1){
            //noop
        }
    }
}
#endif
