#include <rl_tools/operations/cuda.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environment_wrappers/scale_observations/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>



#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#ifdef RL_TOOLS_ENABLE_JSON
#include <nlohmann/json.hpp>
#include <fstream>
#endif

namespace rlt = rl_tools;

#include "../cpu/config.h"

using DEVICE = rlt::devices::DefaultCUDA;
using KERNEL_DEVICE = rlt::devices::cuda::TAG<DEVICE, true>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
static constexpr bool DYNAMIC_ALLOCATION = false;

using CONFIG = CONFIG_FACTORY<DEVICE, TYPE_POLICY, DYNAMIC_ALLOCATION>;

CONFIG::EVAL_BUFFER eval_buffer;


template <typename DEVICE>
__global__ void test(DEVICE& device, CONFIG::LOOP_STATE* ts){
    printf("%d\n", ts->step);
}


int main(int argc, char** argv) {
    DEVICE device;
    CONFIG::LOOP_STATE* ts = nullptr;
    cudaMalloc(&ts, sizeof(CONFIG::LOOP_STATE));
    test<<<1, 1>>>(device, ts);
    cudaDeviceSynchronize();
    return 0;
}

// Should take ~ 0.3s on M3 Pro in BECHMARK mode
// - tested @ 1118e19f904a26a9619fac7b1680643a0afcb695)
// - tested @ 361c2f5e9b14d52ee497139a3b82867fce0404a7
