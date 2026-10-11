#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_CPU_EMSCRIPTEN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_CPU_EMSCRIPTEN_H

#include "cpu.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices{
    namespace cpu_emscripten{
        template <typename T_SPEC>
        struct Specification: T_SPEC{};
    }
    template <typename T_SPEC>
    using CPU_EMSCRIPTEN = CPU<cpu_emscripten::Specification<T_SPEC>>;
    using DefaultCPU_EMSCRIPTEN = CPU_EMSCRIPTEN<DefaultCPUSpecification>;
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
