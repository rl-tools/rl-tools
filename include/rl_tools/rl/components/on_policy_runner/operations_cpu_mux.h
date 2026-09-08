#include "../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_MUX_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_MUX_H

#if defined(RL_TOOLS_BACKEND_ENABLE_CUDA) && defined(RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA)
#include "operations_cuda.h"
#else
#include "operations_cpu.h"
#endif

#endif
