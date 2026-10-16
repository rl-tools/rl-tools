#include "../../../../../version.h"
#include "../../../../../rl_tools.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_CPU_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_RAPTOR_OPERATIONS_CPU_H

#ifdef __CUDACC__
#include "../../../../../nn/layers/gru/helper_operations_cuda.h"
#endif
#include "../../../../../nn/operations_cpu_mux.h"
#include "operations_generic.h"
#include "../../../../../persist/backends/tar/operations_cpu.h"
#include "../../../../../nn/layers/dense/persist.h"
#include "../../../../../nn/layers/gru/persist.h"
#include "../../../../../nn_models/sequential/persist.h"
#include <conta/conta.h>

#include <limits>
#include <string>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    bool load(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, const std::string& path) {
        if constexpr (DEVICE::DEVICE_ID == devices::DeviceId::CUDA) {
            devices::DefaultCPU host;
            rl::environments::hyperdrone::controllers::raptor::Controller<SPEC> staging;
            malloc(host, staging);
            bool loaded;
            try {
                loaded = load(host, staging, path);
                if (loaded) {
                    copy(host, device, staging, controller);
                }
            }
            catch (...) {
                free(host, staging);
                throw;
            }
            free(host, staging);
            return loaded;
        }
        else {
            persist::backends::tar::File<typename SPEC::TI> checkpoint(path, persist::backends::tar::Mode::READ);
            auto actor = get_group(device, checkpoint, "actor");
            return load(device, controller.model, actor);
        }
    }
    template <typename DEVICE, typename SPEC>
    bool init(DEVICE& device, rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>& controller, const std::string& reference = rl::environments::hyperdrone::controllers::raptor::CHECKPOINT) {
        std::string path = reference;
        if (reference.compare(0, 6, "conta:") == 0) {
            std::string error;
            if (!conta::resolve(reference.substr(6), path, error)) {
                return false;
            }
        }
        return load(device, controller, path);
    }
    template <typename SPEC>
    constexpr rl::environments::hyperdrone::controllers::Bounds<typename SPEC::T> command_bounds(const rl::environments::hyperdrone::controllers::raptor::Controller<SPEC>&, typename SPEC::TI) {
        constexpr auto infinity = std::numeric_limits<typename SPEC::T>::infinity();
        return {-infinity, infinity};
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
