#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_CPU_H

#include "devices.h"
#include "../utils/generic/typing.h"

#include <cstddef>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices{
    namespace cpu{
        template <typename T_MATH, typename T_RANDOM, typename T_LOGGING>
        struct Specification{
            using EXECUTION_HINTS = ExecutionHints;
            using MATH = T_MATH;
            using RANDOM = T_RANDOM;
            using LOGGING = T_LOGGING;
            using index_t = size_t;
        };
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::CPU;
            using index_t = size_t;
            static constexpr index_t MAX_INDEX = -1;
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
        using SPEC = T_SPEC;
        using EXECUTION_HINTS = typename SPEC::EXECUTION_HINTS;
        typename SPEC::MATH math;
        typename SPEC::RANDOM random;
        typename SPEC::LOGGING logger;
#ifdef RL_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
        index_t malloc_counter = 0;
#endif
    };

    using DefaultCPUSpecification = cpu::Specification<math::CPU, random::CPU, logging::CPU>;
    using DefaultCPU = CPU<DefaultCPUSpecification>;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEV_SPEC, typename TI>
    void count_malloc(devices::CPU<DEV_SPEC>& device, TI size){
#ifdef RL_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
        device.malloc_counter += size;
#endif
    }
    template <typename SPEC>
    void check_status(devices::CPU<SPEC>& device){ }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
