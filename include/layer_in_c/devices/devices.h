#ifndef LAYER_IN_C_DEVICES_DEVICES_H
#define LAYER_IN_C_DEVICES_DEVICES_H

#include <layer_in_c/utils/generic/typing.h>

namespace layer_in_c {
    namespace devices {
        enum class Device {
            Generic,
            Dummy,
            CPU,
            CPU_BLAS,
            CPU_MKL,
            CPU_ACCELERATE,
            CPU_TENSORBOARD,
            CUDA,
        };
        enum class Type {
            math,
            random,
            logging
        };
    }
}

#endif