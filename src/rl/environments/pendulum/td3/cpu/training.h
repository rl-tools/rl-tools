#ifndef RL_TOOLS_TEST_BARE
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#else
#include <rl_tools/operations/arm.h>
//#include <rl_tools/nn/layers/dense/operations_arm/opt.h>
#include <rl_tools/nn/operations_generic.h>
#endif
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>


#include <rl_tools/rl/algorithms/td3/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/td3/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#include <rl_tools/rl/loop/steps/timing/operations.h>

namespace rlt = rl_tools;

#ifndef RL_TOOLS_TEST_BARE
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
#else
using DEVICE = rlt::devices::DefaultARM;
#endif
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::td3::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    static constexpr TI STEP_LIMIT = 10000;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
};
#if !defined(BENCHMARK) && !defined(RL_TOOLS_TEST_BARE)
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::td3::loop::core::DefaultConfigApproximatorsMLP>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#endif
#if defined(BENCHMARK)
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
#endif
#if defined(RL_TOOLS_TEST_BARE)
using LOOP_CORE_CONFIG = rlt::rl::algorithms::td3::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_EVAL_CONFIG;
#endif
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

TI run(TI seed){
    LOOP_STATE ts;
    rlt::init(ts, seed);
    while(!rlt::step(ts)){
#if !defined(BENCHMARK) && !defined(RL_TOOLS_TEST_BARE)
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
#endif
    }
#if !defined(BENCHMARK)
    auto result = evaluate(ts.device, ts.env_eval, ts.ui, ts.actor_critic.actor, rl_tools::rl::utils::evaluation::Specification<LOOP_EVAL_CONFIG::PARAMETERS::NUM_EVALUATION_EPISODES, LOOP_EVAL_CONFIG::NEXT::PARAMETERS::ENVIRONMENT_STEP_LIMIT>(), ts.observations_mean, ts.observations_std, ts.actor_deterministic_evaluation_buffers, ts.rng, false);
    TI return_code = (TI)(-result.returns_mean / 100);
    rlt::destroy(ts);
    return return_code;
#else
    return 0;
#endif
}
