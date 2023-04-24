#ifndef BACKPROP_TOOLS_DEVICES_CUDA_H
#define BACKPROP_TOOLS_DEVICES_CUDA_H

#include <backprop_tools/utils/generic/typing.h>
#include "devices.h"
#include "cpu.h"
#include <cublas_v2.h>
namespace backprop_tools::devices{
    namespace cuda{
        struct Base{
            static constexpr DeviceId DEVICE_ID = DeviceId::CUDA;
            using index_t = unsigned int;
        };
    }
    namespace math{
        struct CUDA: cuda::Base{
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
        static constexpr bool compatible = OTHER_DEVICE::DEVICE_ID == DeviceId::CUDA;
        using SPEC = T_SPEC;
        typename SPEC::LOGGING* logger = nullptr;
        cublasHandle_t handle;
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
        index_t malloc_counter = 0;
#endif
#ifdef BACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT
        bool initialized = false;
#endif
    };
    struct DefaultCUDASpecification{
//        using MATH_HOST = devices::math::CPU;
        using MATH = devices::math::CUDA;
        using MATH_DEVICE_ACCURATE = math::CUDA;
        using RANDOM = random::CUDA;
        using LOGGING = logging::CUDA;
    };
    using DefaultCUDA = CUDA<DefaultCUDASpecification>;
}

#include <iostream>
namespace backprop_tools {
    template <typename SPEC>
    void init(devices::CUDA<SPEC>& device){
        cublasStatus_t stat;
        stat = cublasCreate(&device.handle);
#ifdef BACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT
        device.initialized = true;
#endif
        if (stat != CUBLAS_STATUS_SUCCESS) {
//            logging::text(device.logger, (const char*)"CUBLAS initialization failed ", cublasGetStatusString(stat));
            std::cout << "CUBLAS initialization failed " << cublasGetStatusString(stat) << std::endl;
        }
    }
    template <typename SPEC>
    void check_status(devices::CUDA<SPEC>& device){
#ifdef BACKPROP_TOOLS_DEBUG_DEVICE_CUDA_CHECK_INIT
        if(!device.initialized){
            std::cerr << "CUDA device not initialized" << std::endl;
        }
#endif
#ifdef BACKPROP_TOOLS_DEBUG_DEVICE_CUDA_SYNCHRONIZE_STATUS_CHECK
        cudaDeviceSynchronize();
#endif
        cudaError_t cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            std::string error_string = cudaGetErrorString(cudaStatus);
            std::cerr << "cuda failed: " << error_string << std::endl;
        }
    }
    template <typename DEV_SPEC, typename TI>
    void count_malloc(devices::CUDA<DEV_SPEC>& device, TI size){
#ifdef BACKPROP_TOOLS_DEBUG_CONTAINER_COUNT_MALLOC
        device.malloc_counter += size;
#endif
    }

}

#endif
