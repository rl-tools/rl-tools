#ifndef BACKPROP_TOOLS_OPERATIONS_CPU_OPENBLAS_GROUP_1
    #define BACKPROP_TOOLS_OPERATIONS_CPU_OPENBLAS_GROUP_1
    #include <cblas.h>
    #include <backprop_tools/devices/cpu_openblas.h>
    #include <backprop_tools/utils/assert/declarations_cpu.h>
    #include <backprop_tools/math/operations_cpu.h>
    #include <backprop_tools/random/operations_cpu.h>
    #include <backprop_tools/logging/operations_cpu.h>
#else
    #error "Group 1 already imported"
#endif
