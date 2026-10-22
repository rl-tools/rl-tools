#pragma once
#include <rl_tools/rl_tools.h>
#include <rl_tools/devices/cpu.h>
#include <rl_tools/logging/operations_cpu.h>

#include <mutex>
#include <stdexcept>

namespace rl_tools::devices::logging {
    struct HyperLux: CPU {};
}

namespace hyperlux {
    using CPUSpecification = rl_tools::devices::cpu::Specification<rl_tools::devices::math::CPU, rl_tools::devices::random::CPU, rl_tools::devices::logging::HyperLux>;

    struct VerbosityState {
        std::mutex mutex;
        rl_tools::devices::logging::Verbosity level = rl_tools::devices::logging::Verbosity::info;
    };

    // Extensions have local symbol visibility: each owns a policy updated by Python.
    inline VerbosityState& verbosity_state(){
        static VerbosityState state;
        return state;
    }

    inline rl_tools::devices::logging::CPU diagnostic_logger(){
        auto& state = verbosity_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        rl_tools::devices::logging::CPU logger;
        logger.verbosity = state.level;
        return logger;
    }

    inline void set_verbosity(int level){
        using Verbosity = rl_tools::devices::logging::Verbosity;
        if(level < (int)Verbosity::error || level > (int)Verbosity::debug){
            throw std::invalid_argument("hyperlux: verbosity level out of range");
        }
        auto& state = verbosity_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.level = (Verbosity)level;
    }
}

namespace rl_tools {
    template <typename DEVICE, typename... ARGS>
    void log(DEVICE& device, devices::logging::HyperLux& logger, devices::logging::Verbosity verbosity, const ARGS&... args){
        auto snapshot = hyperlux::diagnostic_logger();
        log(device, snapshot, verbosity, args...);
    }
}
