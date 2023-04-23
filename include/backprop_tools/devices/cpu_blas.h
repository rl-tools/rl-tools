#ifndef BACKPROP_TOOLS_DEVICES_CPU_BLAS_H
#define BACKPROP_TOOLS_DEVICES_CPU_BLAS_H

#include <backprop_tools/utils/generic/typing.h>
#include "devices.h"

#include "cpu.h"

namespace backprop_tools::devices{
    template <typename T_SPEC>
    struct CPU_BLAS: CPU<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_BLAS;
    };
    using DefaultCPU_BLAS = CPU_BLAS<DefaultCPUSpecification>;
}

#endif
