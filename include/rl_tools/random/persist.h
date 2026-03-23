#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RANDOM_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RANDOM_PERSIST_H
#include "../devices/cpu.h"
#include <sstream>
#include <cstring>
RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    namespace random_persist{
        constexpr auto RNG_STATE_MAX_SIZE = 8192; // mt19937 state is ~5KB
    }
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    void save(devices::CPU<DEV_SPEC>& device, devices::random::CPU::ENGINE<SPEC>& rng, GROUP& group) {
        using TI = typename devices::CPU<DEV_SPEC>::index_t;
        std::stringstream ss;
        ss << rng.engine;
        std::string state_str = ss.str();
        utils::assert_exit(device, state_str.length() < random_persist::RNG_STATE_MAX_SIZE, "RNG state string exceeds maximum size");
        Tensor<tensor::Specification<uint8_t, TI, tensor::Shape<TI, random_persist::RNG_STATE_MAX_SIZE>>> state_tensor;
        malloc(device, state_tensor);
        std::memset(data(state_tensor), 0, random_persist::RNG_STATE_MAX_SIZE);
        std::memcpy(data(state_tensor), state_str.c_str(), state_str.length() + 1); // include null terminator
        auto rng_group = create_group(device, group, "rng");
        save(device, state_tensor, rng_group, "state");
        free(device, state_tensor);
    }
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    bool load(devices::CPU<DEV_SPEC>& device, devices::random::CPU::ENGINE<SPEC>& rng, GROUP& group) {
        using TI = typename devices::CPU<DEV_SPEC>::index_t;
        Tensor<tensor::Specification<uint8_t, TI, tensor::Shape<TI, random_persist::RNG_STATE_MAX_SIZE>>> state_tensor;
        malloc(device, state_tensor);
        auto rng_group = get_group(device, group, "rng");
        if(!load(device, state_tensor, rng_group, "state")){
            free(device, state_tensor);
            return false;
        }
        std::string state_str(reinterpret_cast<const char*>(data(state_tensor)));
        free(device, state_tensor);
        std::stringstream ss(state_str);
        ss >> rng.engine;
        return !ss.fail();
    }
    // ArrayENGINE: save states as a double tensor (double has 53-bit mantissa, exactly represents all uint32 values)
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    void save(devices::CPU<DEV_SPEC>& device, devices::generic::random::ArrayENGINE<SPEC>& rng, GROUP& group) {
        using TI = typename devices::CPU<DEV_SPEC>::index_t;
        constexpr TI NUM_RNGS = SPEC::NUM_RNGS;
        Tensor<tensor::Specification<double, TI, tensor::Shape<TI, NUM_RNGS>>> state_tensor;
        malloc(device, state_tensor);
        for(TI i = 0; i < NUM_RNGS; i++){
            set(device, state_tensor, (double)get(rng.states, 0, i).state, i);
        }
        save(device, state_tensor, group, "states");
        free(device, state_tensor);
    }
    template<typename DEV_SPEC, typename SPEC, typename GROUP>
    bool load(devices::CPU<DEV_SPEC>& device, devices::generic::random::ArrayENGINE<SPEC>& rng, GROUP& group) {
        using TI = typename devices::CPU<DEV_SPEC>::index_t;
        constexpr TI NUM_RNGS = SPEC::NUM_RNGS;
        Tensor<tensor::Specification<double, TI, tensor::Shape<TI, NUM_RNGS>>> state_tensor;
        malloc(device, state_tensor);
        if(!load(device, state_tensor, group, "states")){
            free(device, state_tensor);
            return false;
        }
        for(TI i = 0; i < NUM_RNGS; i++){
            devices::generic::random::PortableState element;
            element.state = static_cast<unsigned int>(get(device, state_tensor, i));
            set(rng.states, 0, i, element);
        }
        free(device, state_tensor);
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
