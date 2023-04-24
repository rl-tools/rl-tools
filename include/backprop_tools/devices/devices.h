#ifndef BACKPROP_TOOLS_DEVICES_DEVICES_H
#define BACKPROP_TOOLS_DEVICES_DEVICES_H

#include <backprop_tools/utils/generic/typing.h>

namespace backprop_tools {
    namespace devices {
        struct ExecutionHints{};
        template <typename DEV_SPEC>
        struct Device{
        };
        // todo: deprecate the global device id and move it to the cpu devices which sometimes need compatibility checks
        enum class DeviceId{
            Generic,
            Dummy,
            CPU,
            CPU_BLAS,
            CPU_MKL,
            CPU_ACCELERATE,
            CPU_TENSORBOARD,
            CUDA,
            ARM,
            ESP32
        };
        enum class Type {
            math,
            random,
            logging
        };
    }
}

namespace backprop_tools{
    template <typename DEV_SPEC>
    void init(devices::Device<DEV_SPEC>& device){ };
    template <typename DEV_SPEC, typename T>
    void count_malloc(devices::Device<DEV_SPEC>& device, T){ };
}

#endif