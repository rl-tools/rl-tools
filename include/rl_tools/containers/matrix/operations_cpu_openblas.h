#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_CONTAINERS_OPERATIONS_CPU_OPENBLAS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_CONTAINERS_OPERATIONS_CPU_OPENBLAS_H

#include "operations_cpu_blas.h"
#include "../containers.h"
#include "../devices/cpu_openblas.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename INPUT_SPEC_A, typename INPUT_SPEC_B, typename OUTPUT_SPEC>
    void multiply(devices::CPU_OPENBLAS<DEV_SPEC>& device, const Matrix<INPUT_SPEC_A>& A, const Matrix<INPUT_SPEC_B>& B, Matrix<OUTPUT_SPEC>& output) {
        multiply((devices::CPU_BLAS<DEV_SPEC>&)device, A, B, output);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
