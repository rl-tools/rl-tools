#include "../../version.h"
#include "../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_MUX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RENDERING_RAYTRACING_OPERATIONS_CPU_MUX_H

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
    #include "backends/metal/operations_cpu.h"
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
    #include "backends/optix/operations_cuda.h"
#else
    #error "RLtools raytracing: no backend configured (RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX or RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)"
#endif

#endif
