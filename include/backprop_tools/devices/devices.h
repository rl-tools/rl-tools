#include "../version.h"
#if !defined(BACKPROP_TOOLS_DEVICES_DEVICES_H) && (BACKPROP_TOOLS_USE_THIS_VERSION == 1)
#define BACKPROP_TOOLS_DEVICES_DEVICES_H

#include <backprop_tools/backprop_tools.h>

#include <backprop_tools/utils/generic/typing.h>

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
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
            CPU_OPENBLAS,
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
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

BACKPROP_TOOLS_NAMESPACE_WRAPPER_START
namespace backprop_tools{
    template <typename DEV_SPEC>
    void init(devices::Device<DEV_SPEC>& device){ };
    template <typename DEV_SPEC, typename T>
    void count_malloc(devices::Device<DEV_SPEC>& device, T){ };
}
BACKPROP_TOOLS_NAMESPACE_WRAPPER_END

#endif