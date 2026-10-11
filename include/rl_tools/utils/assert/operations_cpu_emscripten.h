#include "../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_UTILS_ASSERT_OPERATIONS_CPU_EMSCRIPTEN_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_UTILS_ASSERT_OPERATIONS_CPU_EMSCRIPTEN_H

#include "declarations_cpu_emscripten.h"
#include <cstdlib>
#include <emscripten/em_asm.h>
#include <sstream>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::utils{
    template <typename DEV_SPEC, typename T>
    RL_TOOLS_FUNCTION_PLACEMENT bool assert_exit(devices::CPU_EMSCRIPTEN<DEV_SPEC>& device, bool condition, T message){
        if(!condition){
            log(device, device.logger, message);
            std::ostringstream formatted;
            formatted << message;
            const auto text = formatted.str();
            EM_ASM({ abort(UTF8ToString($0)); }, text.c_str());
            std::abort();
        }
        return condition;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
