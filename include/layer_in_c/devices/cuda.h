#ifndef LAYER_IN_C_DEVICES_CUDA_H
#define LAYER_IN_C_DEVICES_CUDA_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace cuda{
        struct Base{
            static constexpr Device DEVICE = Device::CUDA;
            using index_t = size_t;
        };
    }
    namespace math{
        struct CUDA: cuda::Base{
            static constexpr Domain DOMAIN = Domain::math;
        };
    }
    namespace random{
        struct CUDA: cuda::Base{
            static constexpr Domain DOMAIN = Domain::random;
        };
    }
    namespace logging{
        struct CUDA: cuda::Base{
            static constexpr Domain DOMAIN = Domain::logging;
        };
    }
    template <typename T_SPEC>
    struct CUDA: cuda::Base{
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        explicit CUDA(typename SPEC::LOGGING& logger) : logger(logger) {}
    };
    struct DefaultCUDASpecification{
        using MATH = math::CUDA;
        using RANDOM = random::CUDA;
        using LOGGING = logging::CUDA;
    };
    using DefaultCUDA = CUDA<DefaultCUDASpecification>;
}

#endif
