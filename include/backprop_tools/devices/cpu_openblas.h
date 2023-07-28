#ifndef BACKPROP_TOOLS_DEVICES_CPU_OPENBLAS_H
#define BACKPROP_TOOLS_DEVICES_CPU_OPENBLAS_H

#include <backprop_tools/utils/generic/typing.h>
#include "devices.h"

#include "cpu_blas.h"

#include <cblas.h>

namespace backprop_tools::devices{
    template <typename T_SPEC>
    struct CPU_OPENBLAS: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_OPENBLAS;
    };
    using DefaultCPU_OPENBLAS = CPU_OPENBLAS<DefaultCPUSpecification>;
}
namespace backprop_tools{
    template <typename DEV_SPEC>
    void init(devices::CPU_OPENBLAS<DEV_SPEC>& device){
//        openblas_set_num_threads(4);
    }

}

#endif
