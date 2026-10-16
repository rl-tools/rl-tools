#include "../../../../../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_OPERATIONS_GENERIC_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_CONTROLLERS_DIRECT_OPERATIONS_GENERIC_H

#include "direct.h"
#include "../../../../../containers/tensor/operations_generic.h"
#include "../../../../../containers/matrix/matrix.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE>
    RL_TOOLS_FUNCTION_PLACEMENT bool init(DEVICE&, rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&){
        return true;
    }
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, const rl::environments::hyperdrone::controllers::Direct<SOURCE_SPEC, INTERFACE>&, rl::environments::hyperdrone::controllers::Direct<TARGET_SPEC, INTERFACE>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&){}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, const rl::environments::hyperdrone::controllers::direct::State<SOURCE_SPEC>&, rl::environments::hyperdrone::controllers::direct::State<TARGET_SPEC>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE&, rl::environments::hyperdrone::controllers::direct::Buffer<SPEC>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE&, rl::environments::hyperdrone::controllers::direct::Buffer<SPEC>&){}
    template<typename DEVICE, typename SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void init(DEVICE&, rl::environments::hyperdrone::controllers::direct::Buffer<SPEC>&){}
    template<typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(SOURCE_DEVICE&, TARGET_DEVICE&, const rl::environments::hyperdrone::controllers::direct::Buffer<SOURCE_SPEC>&, rl::environments::hyperdrone::controllers::direct::Buffer<TARGET_SPEC>&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&, RNG&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename MASK_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&, const Tensor<MASK_SPEC>&, RNG&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename MASK_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE&, const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&, rl::environments::hyperdrone::controllers::direct::Buffer<SPEC>&, const Tensor<MASK_SPEC>&, RNG&){}
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename PARAM_SPEC, typename PHYSICAL_SPEC, typename COMMAND_SPEC, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT auto control(DEVICE& device, const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, rl::environments::hyperdrone::controllers::direct::State<SPEC>&, rl::environments::hyperdrone::controllers::direct::Buffer<SPEC>&, const Tensor<PARAM_SPEC>&, const Tensor<PHYSICAL_SPEC>&, const Tensor<COMMAND_SPEC>& commands, RNG&){
        static_assert(product(typename COMMAND_SPEC::SHAPE{}) == SPEC::INSTANCES * 4);
        return view_range(device, commands, tensor::ViewSpec<0, COMMAND_SPEC::SHAPE::template GET<0>>{});
    }
    template<typename DEVICE, typename SPEC, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename T, typename TI, TI N>
    RL_TOOLS_FUNCTION_PLACEMENT bool validate(DEVICE&, const rl::environments::hyperdrone::controllers::Direct<SPEC, INTERFACE>&, const rl::environments::l2f::parameters::Dynamics<T, TI, N>&){
        return true;
    }

    template<typename DEVICE, rl::environments::l2f::parameters::ActionInterface INTERFACE, typename ACTION_SPEC, typename PHYSICAL_STATE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT auto transition_action(DEVICE&, const rl::environments::hyperdrone::controllers::direct::Config<INTERFACE>&, const Matrix<ACTION_SPEC>& command, const PHYSICAL_STATE&, TI agent, TI component){
        return get(command, 0, agent * 4 + component);
    }

}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
