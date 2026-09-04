#include "on_policy_runner.h"
#include "../../environments/batch/operations_generic.h"
#ifndef RL_TOOLS_RL_COMPONENTS_ON_POLICY_RUNNER_OPERATIONS_CPU_DELAY_OPERATIONS_GENERIC_INCLUDE
#ifdef RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include "operations_cuda.h"
#else
#include "operations_generic.h"
#endif
#endif
