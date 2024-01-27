#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>

#include <gtest/gtest.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn/operations_cpu_mux.h>
using DEV_SPEC_INIT = rlt::devices::cpu::Specification<rlt::devices::math::CPU, rlt::devices::random::CPU, rlt::devices::logging::CPU_TENSORBOARD<>>;
using DEVICE_INIT = rlt::devices::CPU<DEV_SPEC_INIT>;
//using DEVICE = rlt::devices::CPU_MKL<DEV_SPEC_INIT>;
using DEVICE = rlt::devices::DefaultCUDA;
using DEV_SPEC = DEVICE::SPEC;

#include <rl_tools/rl/environments/pendulum/operations_generic.h>

#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/rl/components/off_policy_runner/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/operations_cuda.h>
#include <rl_tools/rl/algorithms/sac/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#include <rl_tools/rl/loop/steps/timing/operations.h>



#include <gtest/gtest.h>
#include <filesystem>

using T = float;
using TI = typename DEVICE::index_t;


using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<DEVICE_INIT, T, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::DefaultConfigApproximatorsMLP>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG_INIT = LOOP_TIMING_CONFIG;

using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::DefaultConfigApproximatorsMLP>;

using LOOP_STATE_INIT = LOOP_CONFIG_INIT::State<LOOP_CONFIG_INIT>;
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

LOOP_STATE_INIT ts_init, ts_init2;
LOOP_STATE ts;

TEST(RL_TOOLS_RL_CUDA_TD3, TEST_FULL_TRAINING) {
    rlt::malloc(ts_init);
    rlt::malloc(ts_init2);
    rlt::malloc(ts);
    rlt::init(ts_init);
    rlt::copy(ts_init, ts);
    rlt::copy(ts, ts_init2);
    rlt::copy(ts_init2, ts_init);

    bool finished = false;
    while(!finished){
//        rlt::step(ts_init);
        rlt::step(ts);
    }
    rlt::destroy(ts);
}
