#ifndef LAYER_IN_C_DEVICES_CPU_BLAS_H
#define LAYER_IN_C_DEVICES_CPU_BLAS_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include "cpu.h"

namespace layer_in_c::devices{
    template <typename T_SPEC>
    struct CPU_BLAS: CPU<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_BLAS;
    };
    using DefaultCPU_BLAS = CPU_BLAS<DefaultCPUSpecification>;
}

#endif
