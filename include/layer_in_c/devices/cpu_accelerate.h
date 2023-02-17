#ifndef LAYER_IN_C_DEVICES_CPU_ACCELERATE_H
#define LAYER_IN_C_DEVICES_CPU_ACCELERATE_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include "cpu_blas.h"

namespace layer_in_c::devices{
    template <typename T_SPEC>
    struct CPU_ACCELERATE: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_ACCELERATE;
    };
    using DefaultCPU_ACCELERATE = CPU_ACCELERATE<DefaultCPUSpecification>;
}

#endif
