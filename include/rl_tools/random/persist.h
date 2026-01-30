#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RANDOM_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RANDOM_PERSIST_H
#include "../devices/cpu.h"
#include <sstream>
#include <string>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    void save(devices::CPU<DEV_SPEC>& device, devices::random::CPU::ENGINE<SPEC>& rng, GROUP& group) {
        std::stringstream ss;
        ss << rng.engine;
        std::string state_str = ss.str();
        set_attribute(device, group, "rng_state", state_str.c_str());
        write_attributes(device, group);
    }
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    bool load(devices::CPU<DEV_SPEC>& device, devices::random::CPU::ENGINE<SPEC>& rng, GROUP& group) {
        std::string state_str = get_attribute(device, group, "rng_state");
        std::stringstream ss(state_str);
        ss >> rng.engine;
        return !ss.fail();
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
