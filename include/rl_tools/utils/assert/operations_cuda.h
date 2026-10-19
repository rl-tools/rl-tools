#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_ASSERT_OPERATIONS_CUDA_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_ASSERT_OPERATIONS_CUDA_H
#include <cstdlib>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils{
    template <typename DEV_SPEC, typename T, typename utils::typing::enable_if<!DEV_SPEC::KERNEL && !DEV_SPEC::TAG>::type* = nullptr>
    auto host_assert_exit(devices::CUDA<DEV_SPEC>& dev, bool condition, T message){
        if(!condition){
            log(dev, dev.logger, message);
            std::abort();
        }
        return condition;
    }
    template <typename DEV_SPEC, typename T, typename utils::typing::enable_if<!DEV_SPEC::KERNEL && !DEV_SPEC::TAG>::type* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT bool assert_exit(devices::CUDA<DEV_SPEC>& dev, bool condition, T message){
        return RL_TOOLS_HOST_ONLY(host_assert_exit(dev, condition, message));
    }
    template <typename DEV_SPEC, typename T, typename utils::typing::enable_if<DEV_SPEC::KERNEL || DEV_SPEC::TAG>::type* = nullptr>
    RL_TOOLS_FUNCTION_PLACEMENT bool assert_exit(devices::CUDA<DEV_SPEC>& dev, bool condition, T message){
        //noop for kernel and tag devices
        return condition;
    }
    template <typename SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool assert_exit(devices::cuda::CUDA_KERNEL<SPEC>& dev, bool condition, T message){
        return condition;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif