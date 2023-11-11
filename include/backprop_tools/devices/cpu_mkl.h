#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_CPU_MKL_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_CPU_MKL_H

#include "../utils/generic/typing.h"
#include "devices.h"

#include "cpu_blas.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices{
    template <typename T_SPEC>
    struct CPU_MKL: CPU_BLAS<T_SPEC>{
        static constexpr DeviceId DEVICE_ID = DeviceId::CPU_MKL;
    };
    using DefaultCPU_MKL = CPU_MKL<DefaultCPUSpecification>;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
