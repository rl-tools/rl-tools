#include "../../../../version.h"
#include "../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RAPTOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_RAPTOR_OPERATIONS_CPU_H

#include "operations_generic.h"
#include "../../../../persist/backends/tar/operations_cpu.h"
#include "../../../../nn/layers/dense/persist.h"
#include "../../../../nn/layers/gru/persist.h"
#include "../../../../nn_models/sequential/persist.h"

#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    bool load(DEVICE& device, rl::environments::hyperdrone::Raptor<SPEC>& policy, const std::string& path) {
        persist::backends::tar::File<typename SPEC::TI> checkpoint(path, persist::backends::tar::Mode::READ);
        auto actor = get_group(device, checkpoint, "actor");
        return load(device, policy.model, actor);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
