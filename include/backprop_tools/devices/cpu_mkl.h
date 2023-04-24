#ifndef BACKPROP_TOOLS_DEVICES_CPU_MKL_H
#define BACKPROP_TOOLS_DEVICES_CPU_MKL_H

#include <backprop_tools/utils/generic/typing.h>
#include "devices.h"

#include "cpu_blas.h"

namespace backprop_tools::devices{
    template <typename T_SPEC>
    struct CPU_MKL: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_MKL;
    };
    using DefaultCPU_MKL = CPU_MKL<DefaultCPUSpecification>;
}

#endif
