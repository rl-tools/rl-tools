#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_ASSERT_OPERATIONS_ARM_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_ASSERT_OPERATIONS_ARM_H

#ifdef RL_TOOLS_ARM_TEST
#include <cstdlib>
#include <stdexcept>
#endif
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils{
    template <typename DEV_SPEC, typename T>
    auto host_assert_exit(devices::ARM<DEV_SPEC>& device, bool condition, T message){
        if(!condition){
            log(device, device.logger, message);
#ifdef RL_TOOLS_ARM_TEST
            throw std::runtime_error(message);
#endif
        }
        return condition;
    }
    template <typename DEV_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool assert_exit(devices::ARM<DEV_SPEC>& device, bool condition, T message){
        return RL_TOOLS_HOST_ONLY(host_assert_exit(device, condition, message));
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif