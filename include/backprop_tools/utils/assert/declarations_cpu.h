#ifndef BACKPROP_TOOLS_UTILS_ASSERT_DECLARATIONS_CPU_H
#define BACKPROP_TOOLS_UTILS_ASSERT_DECLARATIONS_CPU_H

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::CPU<DEV_SPEC>& device, bool condition, T message);
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif