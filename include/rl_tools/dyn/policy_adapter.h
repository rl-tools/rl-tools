#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_POLICY_ADAPTER_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_POLICY_ADAPTER_H

#include "model.h"
#include "operations_generic.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::dyn{
    template <typename T_TI>
    struct PolicyState{
        using TI = T_TI;
        State<TI> inner{};
    };
    template <typename T_TI>
    struct PolicyBuffer{
        using TI = T_TI;
        Buffer<TI> inner{};
        Tensor<TensorSpecification<TI>> dyn_input{};
        Tensor<TensorSpecification<TI>> dyn_output{};
    };
    template <typename T_TI, T_TI T_INPUT_DIM, T_TI T_OUTPUT_DIM>
    struct Policy{
        using TI = T_TI;
        static constexpr TI INPUT_DIM = T_INPUT_DIM;
        static constexpr TI OUTPUT_DIM = T_OUTPUT_DIM;
        Layer<TI> layer{};
        using INPUT_SHAPE = tensor::Shape<TI, 1, INPUT_DIM>;
        using OUTPUT_SHAPE = tensor::Shape<TI, 1, OUTPUT_DIM>;
        template <bool DA = true> using State = PolicyState<TI>;
        template <bool DA = true> using Buffer = PolicyBuffer<TI>;
    };
}
namespace rl_tools{ namespace inference{ namespace executor{
    template <typename TI, TI T_INPUT_DIM, TI T_OUTPUT_DIM, typename T_T, TI T_SPEC_INPUT_DIM, bool T_DA>
    struct ObservationTensorType<dyn::Policy<TI, T_INPUT_DIM, T_OUTPUT_DIM>, T_T, TI, T_SPEC_INPUT_DIM, T_DA>{
        using type = dyn::Tensor<dyn::TensorSpecification<TI>>;
    };
}}}
namespace rl_tools{ namespace inference{ namespace applications{ namespace l2f{
    template <typename TI, TI T_INPUT_DIM, TI T_OUTPUT_DIM, typename T_T, TI T_L2F_INPUT_DIM, TI T_L2F_OUTPUT_DIM, bool T_DA>
    struct InputTensorType<dyn::Policy<TI, T_INPUT_DIM, T_OUTPUT_DIM>, T_T, TI, T_L2F_INPUT_DIM, T_L2F_OUTPUT_DIM, T_DA>{
        using type = dyn::Tensor<dyn::TensorSpecification<TI>>;
    };
    template <typename TI, TI T_INPUT_DIM, TI T_OUTPUT_DIM, typename T_T, TI T_L2F_INPUT_DIM, TI T_L2F_OUTPUT_DIM, bool T_DA>
    struct OutputTensorType<dyn::Policy<TI, T_INPUT_DIM, T_OUTPUT_DIM>, T_T, TI, T_L2F_INPUT_DIM, T_L2F_OUTPUT_DIM, T_DA>{
        using type = dyn::Tensor<dyn::TensorSpecification<TI>>;
    };
}}}}
namespace rl_tools{
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::PolicyState<TI>& state){
        rl_tools::malloc(device, state.inner);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::PolicyState<TI>& state){
        rl_tools::free(device, state.inner);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void copy(DEVICE& d1, DEVICE& d2, const dyn::PolicyState<TI>& src, dyn::PolicyState<TI>& dst){
        rl_tools::copy(d1, d2, src.inner, dst.inner);
    }
    template <typename DEVICE, typename TI, TI INPUT_DIM, TI OUTPUT_DIM, typename RNG>
    RL_TOOLS_FUNCTION_PLACEMENT void reset(DEVICE& device, dyn::Policy<TI, INPUT_DIM, OUTPUT_DIM>& policy, dyn::PolicyState<TI>& state, RNG& rng){
        rl_tools::reset(device, policy.layer, state.inner);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void malloc(DEVICE& device, dyn::PolicyBuffer<TI>& buffer){
        rl_tools::malloc(device, buffer.inner);
        rl_tools::malloc(device, buffer.dyn_input);
        rl_tools::malloc(device, buffer.dyn_output);
    }
    template <typename DEVICE, typename TI>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::PolicyBuffer<TI>& buffer){
        rl_tools::free(device, buffer.inner);
        rl_tools::free(device, buffer.dyn_input);
        rl_tools::free(device, buffer.dyn_output);
    }
    template <typename DEVICE, typename TI, TI INPUT_DIM, TI OUTPUT_DIM>
    RL_TOOLS_FUNCTION_PLACEMENT void free(DEVICE& device, dyn::Policy<TI, INPUT_DIM, OUTPUT_DIM>& policy){
        rl_tools::free(device, policy.layer);
    }
    template <typename DEVICE, typename TI, TI INPUT_DIM, TI OUTPUT_DIM, typename OBS_SPEC, typename OUTPUT_SPEC, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, dyn::Policy<TI, INPUT_DIM, OUTPUT_DIM>& policy, Tensor<OBS_SPEC>& observation, dyn::PolicyState<TI>& state, Tensor<OUTPUT_SPEC>& output, dyn::PolicyBuffer<TI>& buffer, RNG& rng, MODE mode){
        for(TI i = 0; i < OBS_SPEC::SHAPE::LAST; i++){
            set(device, buffer.dyn_input, get(device, observation, 0, i), i);
        }
        rl_tools::evaluate_step(device, policy.layer, buffer.dyn_input, state.inner, buffer.dyn_output, buffer.inner);
        for(TI i = 0; i < OUTPUT_SPEC::SHAPE::LAST; i++){
            set(device, output, get(device, buffer.dyn_output, i), 0, i);
        }
    }
    template <typename DEVICE, typename TI, TI INPUT_DIM, TI OUTPUT_DIM, typename RNG, typename MODE>
    RL_TOOLS_FUNCTION_PLACEMENT void evaluate_step(DEVICE& device, dyn::Policy<TI, INPUT_DIM, OUTPUT_DIM>& policy, dyn::Tensor<dyn::TensorSpecification<TI>>& observation, dyn::PolicyState<TI>& state, dyn::Tensor<dyn::TensorSpecification<TI>>& action, dyn::PolicyBuffer<TI>& buffer, RNG& rng, MODE mode){
        rl_tools::evaluate_step(device, policy.layer, observation, state.inner, action, buffer.inner);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
