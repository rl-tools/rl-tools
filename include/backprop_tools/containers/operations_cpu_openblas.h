#ifndef BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_OPENBLAS_H
#define BACKPROP_TOOLS_CONTAINERS_OPERATIONS_CPU_OPENBLAS_H

#include "operations_cpu_blas.h"
#include <backprop_tools/containers.h>
#include <backprop_tools/devices/cpu_openblas.h>

namespace backprop_tools{
    template<typename DEV_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(devices::CPU_OPENBLAS<DEV_SPEC>& device, const Matrix<INPUT_SPEC_A>& A, const Matrix<INPUT_SPEC_B>& B, Matrix<OUTPUT_SPEC>& output) {
        multiply((devices::CPU_BLAS<DEV_SPEC>&)device, A, B, output);
    }
}
#endif
