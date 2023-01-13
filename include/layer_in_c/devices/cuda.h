#ifndef LAYER_IN_C_DEVICES_CUDA_H
#define LAYER_IN_C_DEVICES_CUDA_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include <cstddef>
namespace layer_in_c::devices{
    namespace cuda{
        struct Base{
            static constexpr Device DEVICE = Device::CUDA;
            using index_t = unsigned int;
        };
    }
    namespace math{
        struct CUDA: cuda::Base{
            static constexpr Domain DOMAIN = Domain::math;
        };
        struct CUDA_FAST: cuda::Base{
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
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = utils::typing::is_same_v<OTHER_DEVICE, CUDA<T_SPEC>>;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        explicit CUDA(typename SPEC::LOGGING& logger) : logger(logger) {}
    };
    template <typename T_SPEC>
    struct CUDA_GENERIC: cuda::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = utils::typing::is_same_v<OTHER_DEVICE, CUDA_GENERIC<T_SPEC>> || utils::typing::is_same_v<OTHER_DEVICE, CUDA<T_SPEC>>;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        explicit CUDA_GENERIC(typename SPEC::LOGGING& logger) : logger(logger) {}
    };
    struct DefaultCUDASpecification{
        using MATH = math::CUDA_FAST;
        using RANDOM = random::CUDA;
        using LOGGING = logging::CUDA;
    };
    using DefaultCUDA = CUDA<DefaultCUDASpecification>;
    using DefaultCUDAGeneric = CUDA_GENERIC<DefaultCUDASpecification>;
}

#endif
