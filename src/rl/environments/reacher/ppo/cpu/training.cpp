#define RL_TOOLS_NN_DISABLE_GENERIC_FORWARD_BACKWARD

#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/rl/environments/reacher/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#ifdef RL_TOOLS_BACKEND_ENABLE_MKL
#include <rl_tools/nn/layers/operations_cpu_mkl.h>
#endif
#include <rl_tools/nn/layers/flatten/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

#include <utility>

namespace rlt = rl_tools;

#include "config.h"

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<float>;
using TI = typename DEVICE::index_t;
static constexpr bool DYNAMIC_ALLOCATION = true;

using CONFIG = CONFIG_FACTORY<DEVICE, TYPE_POLICY, DYNAMIC_ALLOCATION>;

using LOOP_CONFIG = CONFIG::LOOP_TIMING_CONFIG;
using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

static constexpr TI NUM_EPISODES_FINAL_EVAL = 1000;
using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<TYPE_POLICY, TI, typename LOOP_CONFIG::ENVIRONMENT_EVALUATION, NUM_EPISODES_FINAL_EVAL, CONFIG::ENVIRONMENT::EPISODE_STEP_LIMIT>;

using POLICY = rlt::utils::typing::remove_reference_t<decltype(rlt::get_actor(std::declval<LOOP_STATE>()))>;
using EVAL_BUFFER = rlt::rl::utils::evaluation::PolicyBuffer<rlt::rl::utils::evaluation::PolicyBufferSpecification<EVAL_SPEC, POLICY, DYNAMIC_ALLOCATION>>;

EVAL_BUFFER eval_buffer;

auto run(TI seed, bool verbose){
    DEVICE device;
    if(verbose){
        rlt::log(device, CONFIG::LOOP_TIMING_CONFIG{});
    }
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::malloc(device, eval_buffer);
    rlt::init(device, ts, seed);
    while(!rlt::step(device, ts)){
    }
    rlt::log(device, device.logger, "PPO steps: ", ts.step);
    rlt::rl::utils::evaluation::Result<EVAL_SPEC> result;
    auto actor = rlt::get_actor(ts);
    auto& env = rlt::get_ref(device, ts.envs, 0);
    evaluate(device, env, ts.ui, actor, eval_buffer, result, ts.rng, rlt::Mode<rlt::mode::Evaluation<>>{});
    rlt::log(device, device.logger, "Final return: ", result.returns_mean);
    rlt::log(device, device.logger, "              mean: ", result.returns_mean);
    rlt::log(device, device.logger, "              std : ", result.returns_std);
    rlt::free(device, ts);
    rlt::free(device, eval_buffer);
    return result;
}

int main(int argc, char** argv){
    bool verbose = true;
    std::vector<decltype(run(0, verbose))> returns;
    for(TI seed = 0; seed < 1; seed++){
        auto return_stats = run(seed, verbose);
        returns.push_back(return_stats);
    }
    T sum = 0;
    T sum_squared = 0;
    for(auto& return_stats: returns){
        sum += return_stats.returns_mean;
        sum_squared += return_stats.returns_mean * return_stats.returns_mean;
    }
    T mean = sum / returns.size();
    T std_val = std::sqrt(sum_squared / returns.size() - mean * mean);
    std::sort(returns.begin(), returns.end(), [](auto& a, auto& b){
        return a.returns_mean < b.returns_mean;
    });
    T median = returns[returns.size() / 2].returns_mean;
    std::cout << "Mean return: " << mean << std::endl;
    std::cout << "Std return: " << (returns.size() > 1 ? std::to_string(std_val) : "-") << std::endl;
    std::cout << "Median return: " << median << std::endl;
    return 0;
}
