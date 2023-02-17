#ifndef LAYER_IN_C_DEVICES_CPU_MKL_H
#define LAYER_IN_C_DEVICES_CPU_MKL_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include "cpu_blas.h"

namespace layer_in_c::devices{
    template <typename T_SPEC>
    struct CPU_MKL: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_MKL;
    };
    using DefaultCPU_MKL = CPU_MKL<DefaultCPUSpecification>;
}

#endif
