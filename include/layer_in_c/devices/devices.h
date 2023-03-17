#ifndef LAYER_IN_C_DEVICES_DEVICES_H
#define LAYER_IN_C_DEVICES_DEVICES_H

#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c {
    namespace devices {
        struct ExecutionHints{};
        template <typename DEV_SPEC>
        struct Device{
        };
        enum class DeviceId{
            Generic,
            Dummy,
            CPU,
            CPU_BLAS,
            CPU_MKL,
            CPU_ACCELERATE,
            CPU_TENSORBOARD,
            CUDA,
            ARM
        };
        enum class Type {
            math,
            random,
            logging
        };
    }
}

namespace layer_in_c{
    template <typename DEV_SPEC>
    void init(devices::Device<DEV_SPEC>& device){
    };
}

#endif