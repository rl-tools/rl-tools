#ifndef LAYER_IN_C_DEVICES_CPU_H
#define LAYER_IN_C_DEVICES_CPU_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace cpu{
        template <typename T_MATH, typename T_RANDOM, typename T_LOGGING>
        struct Specification{
            using EXECUTION_HINTS = ExecutionHints;
            using MATH = T_MATH;
            using RANDOM = T_RANDOM;
            using LOGGING = T_LOGGING;
        };
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::CPU;
            using index_t = size_t;
        };
    }
    namespace math{
        struct CPU: cpu::Base{
            static constexpr Type TYPE = Type::math;
        };
    }
    namespace random{
        struct CPU: cpu::Base{
            static constexpr Type TYPE = Type::random;
        };
    }
    namespace logging{
        struct CPU: cpu::Base{
            static constexpr Type TYPE = Type::logging;
        };
    }
    template <typename T_SPEC>
    struct CPU: Device<T_SPEC>, cpu::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible =
            OTHER_DEVICE::DEVICE == DeviceId::Dummy ||
            OTHER_DEVICE::DEVICE == DeviceId::CPU ||
            OTHER_DEVICE::DEVICE == DeviceId::CPU_BLAS ||
            OTHER_DEVICE::DEVICE == DeviceId::CPU_MKL ||
            OTHER_DEVICE::DEVICE == DeviceId::CPU_ACCELERATE ||
            OTHER_DEVICE::DEVICE == DeviceId::CPU_TENSORBOARD;
        using SPEC = T_SPEC;
        using EXECUTION_HINTS = typename SPEC::EXECUTION_HINTS;
        typename SPEC::LOGGING* logger = nullptr;
#ifdef LAYER_IN_C_DEBUG_CONTAINER_COUNT_MALLOC
        index_t malloc_counter = 0;
#endif
    };

    using DefaultCPUSpecification = cpu::Specification<math::CPU, random::CPU, logging::CPU>;
    using DefaultCPU = CPU<DefaultCPUSpecification>;
}

namespace layer_in_c{
    template <typename DEV_SPEC, typename TI>
    void count_malloc(devices::CPU<DEV_SPEC>& device, TI size){
#ifdef LAYER_IN_C_DEBUG_CONTAINER_COUNT_MALLOC
        device.malloc_counter += size;
#endif
    }
}

#endif
