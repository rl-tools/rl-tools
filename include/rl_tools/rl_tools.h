#include "version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_TOOLS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_TOOLS_H
//#define RL_TOOLS_NAMESPACE_WRAPPER test_ns
#ifndef RL_TOOLS_NAMESPACE_WRAPPER
#define RL_TOOLS_NAMESPACE_WRAPPER
#define RL_TOOLS_NAMESPACE_WRAPPER_START
#define RL_TOOLS_NAMESPACE_WRAPPER_END
#else
#define RL_TOOLS_NAMESPACE RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools
#define RL_TOOLS_NAMESPACE_WRAPPER_START namespace RL_TOOLS_NAMESPACE_WRAPPER {
#define RL_TOOLS_NAMESPACE_WRAPPER_END }
#endif

// Execution-space contract: RL_TOOLS_FUNCTION_PLACEMENT promises that every callee of the function is available in every execution space the function is instantiated for.
// Backend leaves that need host facilities (kernel launches, cuBLAS/cuDNN, memcpy, host RNG, standard I/O) keep the placement but wrap their single statement in RL_TOOLS_HOST_ONLY:
// the statement in the host pass and a call to the undefined device sentinel in device passes, so a kernel reaching them fails at device link time instead of compiling silently.
// Tag-device leaves mirror this with RL_TOOLS_DEVICE_ONLY and the host sentinel. This is the only place that tests the compilation pass.
#ifndef RL_TOOLS_FUNCTION_PLACEMENT
    #if defined(__CUDACC__)
        #define RL_TOOLS_FUNCTION_PLACEMENT __host__ __device__
    #else
        #define RL_TOOLS_FUNCTION_PLACEMENT
    #endif
#endif
// The sentinel is instantiated with decltype of the guarded expression: naming the expression keeps it unevaluated (no cross-execution-space call), yet forces the
// host implementation to be instantiated when it has a deduced return type, which is what makes the device pass see the kernel template instantiations inside it.
#if defined(__CUDA_ARCH__)
    #define RL_TOOLS_HOST_ONLY(...) RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::detail::host_only_operation_called_from_device<decltype(__VA_ARGS__)>()
    #define RL_TOOLS_DEVICE_ONLY(...) (__VA_ARGS__)
#else
    #define RL_TOOLS_HOST_ONLY(...) (__VA_ARGS__)
    #define RL_TOOLS_DEVICE_ONLY(...) RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools::detail::device_only_operation_called_from_host<decltype(__VA_ARGS__)>()
#endif
#if defined(__CUDACC_RDC__)
#include <cstdio>
#endif

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace detail{
#if defined(__CUDACC__)
    #if defined(__CUDACC_RDC__)
        // relocatable device code resolves undefined device symbols at module load and fails silently, so trap loudly instead
        template <typename T>
        [[noreturn]] __device__ inline T host_only_operation_called_from_device(){
            printf("RLtools: host-only operation reached from device code\n");
            while(true){
                __trap();
            }
        }
    #else
        template <typename T>
        [[noreturn]] __device__ T host_only_operation_called_from_device();
    #endif
#endif
        template <typename T>
        [[noreturn]] T device_only_operation_called_from_host();
    }
    template <auto T_C>
    struct Constant{
        static constexpr auto C = T_C;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#if defined(_MSC_VER)
    #define RL_TOOLS_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
    #define RL_TOOLS_RESTRICT __restrict__
#else
    #define RL_TOOLS_RESTRICT
#endif


#endif

//#define RL_TOOLS_TARGET_COMMIT_HASH 111f0b82ba87ee4f1938f4d85ee2d266e70f4498
//#define RL_TOOLS_TARGET_COMMIT_HASH_SHORT 1122059
