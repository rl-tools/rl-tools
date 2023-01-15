#ifndef LAYER_IN_C_DEVICES_CPU_MKL_H
#define LAYER_IN_C_DEVICES_CPU_MKL_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include "cpu.h"

namespace layer_in_c::devices{
    template <typename T_SPEC>
    struct CPU_MKL: CPU<T_SPEC>{
        explicit CPU_MKL(typename T_SPEC::LOGGING& logger) : CPU<T_SPEC>(logger) {}
    };
    using DefaultCPU_MKL = CPU_MKL<DefaultCPUSpecification>;
}

#endif
