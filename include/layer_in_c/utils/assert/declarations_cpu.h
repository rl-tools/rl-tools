#ifndef LAYER_IN_C_UTILS_ASSERT_DECLARATIONS_CPU_H
#define LAYER_IN_C_UTILS_ASSERT_DECLARATIONS_CPU_H

namespace layer_in_c::utils{
    template <typename DEV_SPEC, typename T>
    void assert_exit(devices::CPU<DEV_SPEC>& device, bool condition, T message);
}

#endif