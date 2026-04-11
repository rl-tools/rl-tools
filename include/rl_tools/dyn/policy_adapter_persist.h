#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DYN_POLICY_ADAPTER_PERSIST_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DYN_POLICY_ADAPTER_PERSIST_H

#include "policy_adapter.h"
#include "persist.h"

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools{
    template <typename DEVICE, typename SPEC, typename TI, TI INPUT_DIM, TI OUTPUT_DIM, typename GROUP>
    RL_TOOLS_FUNCTION_PLACEMENT bool load(DEVICE& device, inference::applications::L2F<SPEC>& l2f, dyn::Policy<TI, INPUT_DIM, OUTPUT_DIM>& policy, GROUP& actor_group){
        if(!rl_tools::load(device, policy.layer, actor_group)){
            return false;
        }
        TI input_dim = dyn::infer_flat_input_dim(policy.layer);
        if(input_dim == 0){
            return false;
        }
        TI input_shape[] = {1, input_dim};
        dyn::propagate_shapes(policy.layer, input_shape, (TI)2);
        if(policy.layer.output_size == 0){
            return false;
        }
        TI output_dim = policy.layer.output_shape[policy.layer.output_rank - 1];
        if(output_dim != OUTPUT_DIM){
            return false;
        }
        auto& executor = l2f.executor;
        executor.policy_state.inner.batch_size = 1;
        executor.policy_state.inner.layer = &policy.layer;
        executor.policy_state_temp.inner.batch_size = 1;
        executor.policy_state_temp.inner.layer = &policy.layer;
        executor.policy_buffer.inner.layer = &policy.layer;
        dyn::set_shape(executor.policy_buffer.dyn_input, (TI)2, input_shape);
        executor.policy_buffer.dyn_input.type = dyn::Type::FLOAT32;
        TI output_shape[] = {1, output_dim};
        dyn::set_shape(executor.policy_buffer.dyn_output, (TI)2, output_shape);
        executor.policy_buffer.dyn_output.type = dyn::Type::FLOAT32;
        executor.policy_buffer.dyn_output.capacity = policy.layer.output_size;
        dyn::set_shape(executor.observation, (TI)2, input_shape);
        executor.observation.type = dyn::Type::FLOAT32;
        dyn::set_shape(l2f.input, (TI)2, input_shape);
        l2f.input.type = dyn::Type::FLOAT32;
        dyn::set_shape(l2f.output, (TI)2, output_shape);
        l2f.output.type = dyn::Type::FLOAT32;
        rl_tools::malloc(device, l2f);
        return true;
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END
#endif
