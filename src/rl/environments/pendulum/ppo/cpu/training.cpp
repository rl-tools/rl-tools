//#define RL_TOOLS_BACKEND_DISABLE_BLAS

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environment_wrappers/scale_observations/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
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

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using PRE_ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
using SCALE_OBSERVATIONS_WRAPPER_SPEC = rlt::rl::environment_wrappers::scale_observations::Specification<T, TI>;
using ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_OBSERVATIONS_WRAPPER_SPEC, PRE_ENVIRONMENT>;
enum class BENCHMARK_MODE: TI{
    LARGE = 0,
    MEDIUM = 1,
    SMALL = 2,
    TINY = 3,
};
template <typename MODE>
auto constexpr name(MODE mode){
    if(mode == BENCHMARK_MODE::LARGE){
        return "LARGE";
    }else if(mode == BENCHMARK_MODE::MEDIUM){
        return "MEDIUM";
    }else if(mode == BENCHMARK_MODE::SMALL){
        return "SMALL";
    }else if(mode == BENCHMARK_MODE::TINY){
        return "TINY";
    }else{
        return "UNKNOWN";
    }
}

template <BENCHMARK_MODE MODE>
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = MODE == BENCHMARK_MODE::LARGE ? 5 : (MODE == BENCHMARK_MODE::TINY ? 1 : 2);
        static constexpr T GAMMA = 0.9;
        static constexpr T INITIAL_ACTION_STD = 2.0;
        static constexpr bool NORMALIZE_OBSERVATIONS = true;
    };
    static constexpr TI BATCH_SIZE = MODE == BENCHMARK_MODE::LARGE ? 512 : (MODE == BENCHMARK_MODE::MEDIUM ? 256 : (MODE == BENCHMARK_MODE::SMALL ? 64 : (MODE == BENCHMARK_MODE::TINY ? 32 : 64)));
    static constexpr TI ACTOR_HIDDEN_DIM = MODE == BENCHMARK_MODE::LARGE ? 256 : (MODE == BENCHMARK_MODE::MEDIUM ? 64 : (MODE == BENCHMARK_MODE::SMALL ? 32 : (MODE == BENCHMARK_MODE::TINY ? 16 : 64)));
    static constexpr TI CRITIC_HIDDEN_DIM = MODE == BENCHMARK_MODE::LARGE ? 256 : (MODE == BENCHMARK_MODE::MEDIUM ? 64 : (MODE == BENCHMARK_MODE::SMALL ? 32 : (MODE == BENCHMARK_MODE::TINY ? 16 : 64)));
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = MODE == BENCHMARK_MODE::MEDIUM ? 1024 : 256;
    static constexpr TI N_ENVIRONMENTS = MODE == BENCHMARK_MODE::LARGE ? 16 : 4;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
    using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<T>;
};
template <BENCHMARK_MODE MODE>
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS<MODE>, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 10;
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};

template <BENCHMARK_MODE MODE, TI NUM_EPISODES_FINAL_EVAL>
auto run(TI seed, bool verbose){
    DEVICE device;
#ifndef RL_TOOLS_RL_ENVIRONMENTS_PENDULUM_PPO_BENCHMARK
    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG<MODE>, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG<MODE>>>;
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
#else
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG<MODE>>;
#endif
    if(verbose){
        std::cout << "Benchmark mode: " << name(MODE) << std::endl;
        rlt::log(device, LOOP_TIMING_CONFIG{});
    }
    using LOOP_CONFIG = LOOP_TIMING_CONFIG;
    using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    while(!rlt::step(device, ts)){
    }
    using RESULT_SPEC = rlt::rl::utils::evaluation::Specification<T, TI, typename LOOP_CONFIG::ENVIRONMENT_EVALUATION, NUM_EPISODES_FINAL_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;
    rlt::rl::utils::evaluation::Result<RESULT_SPEC> result;
    evaluate(device, ts.envs[0], ts.ui, rlt::get_actor(ts), result, ts.actor_deterministic_evaluation_buffers, ts.rng, false);
    rlt::log(device, device.logger, "Final return: ", result.returns_mean);
    rlt::log(device, device.logger, "              mean: ", result.returns_mean);
    rlt::log(device, device.logger, "              std : ", result.returns_std);
    return result;
}

static constexpr TI NUM_EPISODES_FINAL_EVAL = 1000;
auto run(TI seed, bool verbose, BENCHMARK_MODE mode){
    if(mode == BENCHMARK_MODE::LARGE){
        return run<BENCHMARK_MODE::LARGE, NUM_EPISODES_FINAL_EVAL>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::MEDIUM){
        return run<BENCHMARK_MODE::MEDIUM, NUM_EPISODES_FINAL_EVAL>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::SMALL){
        return run<BENCHMARK_MODE::SMALL, NUM_EPISODES_FINAL_EVAL>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::TINY){
        return run<BENCHMARK_MODE::TINY, NUM_EPISODES_FINAL_EVAL>(seed, verbose);
    }else{
        std::cout << "Unknown benchmark mode: " << static_cast<TI>(mode) << std::endl;
        return decltype(run<BENCHMARK_MODE::TINY, NUM_EPISODES_FINAL_EVAL>(seed, verbose)){};
    }
}


//int main(int argc, char** argv) {
//    TI seed = 0;
//    if (argc > 1) {
//        seed = std::atoi(argv[1]);
//    }
//    BENCHMARK_MODE mode = BENCHMARK_MODE::MEDIUM;
//    if (argc > 2) {
//        mode = static_cast<BENCHMARK_MODE>(std::atoi(argv[2]));
//    }
//    bool verbose = true;
//    if (argc > 3) {
//        verbose = std::atoi(argv[3]);
//    }
//    run(seed, verbose, mode);
//    return 0;
//}

int main(int argc, char** argv) {
    BENCHMARK_MODE mode = BENCHMARK_MODE::MEDIUM;
    bool verbose = true;
    std::vector<decltype(run(0, verbose, mode))> returns;
    for (TI seed=0; seed < 1; seed++){
        auto return_stats = run(seed, verbose, mode);
        returns.push_back(return_stats);
    }
    T sum = 0;
    T sum_squared = 0;
    for(auto& return_stats: returns){
        sum += return_stats.returns_mean;
        sum_squared += return_stats.returns_mean * return_stats.returns_mean;
    }
    T mean = sum / returns.size();
    T std = std::sqrt(sum_squared / returns.size() - mean * mean);
    // median
    std::sort(returns.begin(), returns.end(), [](auto& a, auto& b){
        return a.returns_mean < b.returns_mean;
    });
    T median = returns[returns.size() / 2].returns_mean;
    std::cout << "Mean return: " << mean << std::endl;
    std::cout << "Std return: " << std << std::endl;
    std::cout << "Median return: " << median << std::endl;
#ifdef RL_TOOLS_ENABLE_JSON
    nlohmann::json j;
    for(auto& return_stats: returns){
        j.push_back(return_stats.returns);
    }
    std::ofstream file("pendulum_ppo_returns.json");
    file << j.dump(4);
#endif
    return 0;
}
