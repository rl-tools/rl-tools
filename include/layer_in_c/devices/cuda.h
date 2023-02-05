#ifndef LAYER_IN_C_DEVICES_CUDA_H
#define LAYER_IN_C_DEVICES_CUDA_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"
#include "cpu.h"
#include <cublas_v2.h>
namespace layer_in_c::devices{
    namespace cuda{
        struct Base{
            static constexpr Device DEVICE = Device::CUDA;
            using index_t = unsigned int;
        };
    }
    namespace math{
        struct CUDA: cuda::Base{
            static constexpr Type TYPE = Type::math;
        };
        struct CUDA_FAST: cuda::Base{
            static constexpr Type TYPE = Type::math;
        };
    }
    namespace random{
        struct CUDA: cuda::Base{
            static constexpr Type TYPE = Type::random;
        };
    }
    namespace logging{
        struct CUDA: cuda::Base{
            static constexpr Type TYPE = Type::logging;
        };
    }
    template <typename T_SPEC>
    struct CUDA: cuda::Base{
        template <typename OTHER_DEVICE>
        static constexpr bool compatible = utils::typing::is_same_v<OTHER_DEVICE, CUDA<T_SPEC>>;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING& logger;
        cublasHandle_t handle;
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
        using MATH = devices::math::CPU;
        using MATH_DEVICE = math::CUDA_FAST;
        using MATH_DEVICE_ACCURATE = math::CUDA;
        using RANDOM = random::CUDA;
        using LOGGING = logging::CUDA;
    };
    using DefaultCUDA = CUDA<DefaultCUDASpecification>;
    using DefaultCUDAGeneric = CUDA_GENERIC<DefaultCUDASpecification>;
}

namespace layer_in_c {
    template <typename SPEC>
    void init(devices::CUDA<SPEC>& device){
        cublasStatus_t stat;
        stat = cublasCreate(&device.handle);
        if (stat != CUBLAS_STATUS_SUCCESS) {
//            logging::text(device.logger, (const char*)"CUBLAS initialization failed ", cublasGetStatusString(stat));
            std::cout << "CUBLAS initialization failed " << cublasGetStatusString(stat) << std::endl;
        }
    }
}

#endif
