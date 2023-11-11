#include "../version.h"
#if (defined(BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(BACKPROP_TOOLS_UTILS_LOGGING_OPERATIONS_CUDA_H)) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define BACKPROP_TOOLS_UTILS_LOGGING_OPERATIONS_CUDA_H


#include "../devices/cuda.h"

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace logging{
        template <typename DEVICE, typename A>
        void text(DEVICE& device, devices::logging::CUDA* logger, const char * a, const char * b){
            std::cout << a << b << std::endl;
        }
        template <typename DEVICE, typename A>
        void text(DEVICE& device, devices::logging::CUDA* logger, const A a){
        }
        template <typename DEVICE, typename A, typename B>
        void text(DEVICE& device, devices::logging::CUDA* logger, const A a, const B b){
        }
        template <typename DEVICE, typename A, typename B, typename C, typename D>
        void text(DEVICE& device, devices::logging::CUDA* logger, const A a, const B b, const C c, const D d){
        }
        template <typename DEVICE>
        void set_step(DEVICE& device, devices::logging::CUDA* logger, typename DEVICE::index_t step){ /* noop */ }
        template <typename DEVICE, typename ARG_1, typename ARG_2>
        void construct(DEVICE& device, devices::logging::CUDA* logger, ARG_1, ARG_2){ /* noop */ }
        template <typename DEVICE>
        void construct(DEVICE& device, devices::logging::CUDA* logger){ /* noop */ }
        template <typename DEVICE>
        void destruct(DEVICE& device, devices::logging::CUDA* logger){ /* noop */ }
        template <typename DEVICE, typename TOPIC, typename ARG>
        void add_scalar(DEVICE& device, devices::logging::CUDA* logger, const TOPIC, const ARG){ /* noop */ }
        template <typename DEVICE, typename TOPIC, typename ARG, typename CADENCE>
        void add_scalar(DEVICE& device, devices::logging::CUDA* logger, const TOPIC, const ARG, const CADENCE){ /* noop */ }
        template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN, typename CADENCE>
        void add_histogram(DEVICE& device, devices::logging::CUDA* logger, const TOPIC, const ARG*, const ARG_LEN, const CADENCE){ /* noop */ }
        template <typename DEVICE, typename TOPIC, typename ARG, typename ARG_LEN>
        void add_histogram(DEVICE& device, devices::logging::CUDA* logger, const TOPIC, const ARG*, const ARG_LEN){ /* noop */ }
    }
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END
#endif
