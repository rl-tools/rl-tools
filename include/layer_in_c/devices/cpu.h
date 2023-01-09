#ifndef LAYER_IN_C_DEVICES_CPU_H
#define LAYER_IN_C_DEVICES_CPU_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace cpu{
        struct Base{
            static constexpr Device DEVICE = Device::CPU;
            using index_t = size_t;
        };
    }
    namespace math{
        struct CPU: cpu::Base{
            static constexpr Domain DOMAIN = Domain::math;
        };
    }
    namespace random{
        struct CPU: cpu::Base{
            static constexpr Domain DOMAIN = Domain::random;
        };
    }
    namespace logging{
        struct CPU: cpu::Base{
            static constexpr Domain DOMAIN = Domain::logging;
        };
        struct CPU_WANDB: cpu::Base{
            static constexpr Domain DOMAIN = Domain::logging;
            int counter = 0;
        };
    }
    template <typename T_SPEC>
    struct CPU: cpu::Base{
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        explicit CPU(typename SPEC::LOGGING& logger) : logger(logger) {}
    };
    struct DefaultCPUSpecification{
        using MATH = math::CPU;
        using RANDOM = random::CPU;
        using LOGGING = logging::CPU;
    };
    using DefaultCPU = CPU<DefaultCPUSpecification>;
}

#endif
