#ifndef LAYER_IN_C_DEVICES_CUBLAS_H
#define LAYER_IN_C_DEVICES_CUBLAS_H

#include <layer_in_c/utils/generic/typing.h>
#include "devices.h"

#include "cpu.h"

namespace layer_in_c::devices{
    template <typename T_SPEC>
    struct CUBLAS: CPU<T_SPEC>{
        explicit CUBLAS(typename T_SPEC::LOGGING& logger) : CPU<T_SPEC>(logger) {}
    };
    using DefaultCUBLAS = CUBLAS<DefaultCPUSpecification>;
}

#endif
