#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RANDOM_OPERATIONS_GENERIC_ARRAY_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RANDOM_OPERATIONS_GENERIC_ARRAY_H

#include "../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices::generic::random{
    // PortableState and ArrayENGINE forward-declared in devices/devices.h
    template <typename T_TI, T_TI T_NUM_RNGS, bool T_DYNAMIC_ALLOCATION>
    struct ArraySpecification{
        using TI = T_TI;
        static constexpr TI NUM_RNGS = T_NUM_RNGS;
        static constexpr bool DYNAMIC_ALLOCATION = T_DYNAMIC_ALLOCATION;
    };
    template <typename T_SPEC>
    struct ArrayENGINE{
        using SPEC = T_SPEC;
        using TI = typename SPEC::TI;
        static constexpr TI NUM_RNGS = SPEC::NUM_RNGS;
        Matrix<matrix::Specification<PortableState, TI, 1, NUM_RNGS, SPEC::DYNAMIC_ALLOCATION>> states;
    };
}
RL_TOOLS_NAMESPACE_WRAPPER_END

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, devices::generic::random::ArrayENGINE<SPEC>& rng){
        malloc(device, rng.states);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, devices::generic::random::ArrayENGINE<SPEC>& rng){
        free(device, rng.states);
    }
    template <typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE& device, devices::generic::random::ArrayENGINE<SPEC>& rng, unsigned int seed = 1){
        using TI = typename SPEC::TI;
        for(TI i = 0; i < SPEC::NUM_RNGS; i++){
            devices::generic::random::PortableState element;
            element.state = 0b10101010101010101010101010101010 + seed + static_cast<unsigned int>(i);
            set(rng.states, 0, i, element);
        }
    }
    template <typename DEVICE_SRC, typename DEVICE_DST, typename SPEC>
    void copy(DEVICE_SRC& device_src, DEVICE_DST& device_dst, const devices::generic::random::ArrayENGINE<SPEC>& src, devices::generic::random::ArrayENGINE<SPEC>& dst){
        copy(device_src, device_dst, src.states, dst.states);
    }
    template <typename DEVICE, typename SPEC>
    typename DEVICE::index_t abs_diff(DEVICE& device, devices::generic::random::ArrayENGINE<SPEC>& a, devices::generic::random::ArrayENGINE<SPEC>& b){
        using TI = typename DEVICE::index_t;
        TI acc = 0;
        for(TI i = 0; i < SPEC::NUM_RNGS; i++){
            auto sa = get(a.states, 0, i);
            auto sb = get(b.states, 0, i);
            if(sa.state != sb.state) acc++;
        }
        return acc;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
