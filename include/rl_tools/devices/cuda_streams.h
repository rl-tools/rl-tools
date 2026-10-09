#include "../version.h"
#if (defined(RL_TOOLS_DISABLE_INCLUDE_GUARDS) || !defined(RL_TOOLS_DEVICES_CUDA_STREAMS_H)) && (RL_TOOLS_USE_THIS_VERSION == 1)
#pragma once
#define RL_TOOLS_DEVICES_CUDA_STREAMS_H

#include "../rl_tools.h"
#include <cuda_runtime.h>

RL_TOOLS_NAMESPACE_WRAPPER_START
namespace rl_tools::devices::cuda{
    // orders consumer after the work enqueued so far on producer; the wait captures the record on
    // the spot, so one event serves every join
    inline void order_after(cudaEvent_t event, cudaStream_t producer, cudaStream_t consumer){
        if(producer == consumer){
            return;
        }
        cudaEventRecord(event, producer);
        cudaStreamWaitEvent(consumer, event, 0);
    }
}
RL_TOOLS_NAMESPACE_WRAPPER_END

#endif
