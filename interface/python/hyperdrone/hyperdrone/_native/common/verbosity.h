#pragma once
// Each extension module is loaded with local symbol visibility, so the level lives per module:
// Python pushes it into every loaded module, which forwards it to the loggers of its devices.
#include <rl_tools/devices/cpu.h>

#include <set>
#include <stdexcept>

namespace hyperdrone {
    struct VerbosityRegistry {
        rl_tools::devices::logging::Verbosity verbosity = rl_tools::devices::logging::Verbosity::info;
        std::set<rl_tools::devices::logging::CPU*> loggers;
    };

    inline VerbosityRegistry& verbosity_registry(){
        static VerbosityRegistry registry;
        return registry;
    }

    struct VerbosityRegistration {
        rl_tools::devices::logging::CPU* logger;
        explicit VerbosityRegistration(rl_tools::devices::logging::CPU& logger): logger(&logger){
            logger.verbosity = verbosity_registry().verbosity;
            verbosity_registry().loggers.insert(&logger);
        }
        ~VerbosityRegistration(){
            verbosity_registry().loggers.erase(logger);
        }
        VerbosityRegistration(const VerbosityRegistration&) = delete;
        VerbosityRegistration& operator=(const VerbosityRegistration&) = delete;
    };

    inline void set_verbosity(int level){
        using Verbosity = rl_tools::devices::logging::Verbosity;
        if(level < (int)Verbosity::error || level > (int)Verbosity::debug){
            throw std::invalid_argument("hyperdrone: verbosity level out of range");
        }
        auto& registry = verbosity_registry();
        registry.verbosity = (Verbosity)level;
        for(auto* logger : registry.loggers){
            logger->verbosity = registry.verbosity;
        }
    }
}
