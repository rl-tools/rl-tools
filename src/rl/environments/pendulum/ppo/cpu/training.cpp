//#define RL_TOOLS_BACKEND_DISABLE_BLAS

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>


#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>
#include <rl_tools/rl/loop/steps/timing/operations_cpu.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
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
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::Parameters<T, TI, ENVIRONMENT>{
    struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
        static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
        static constexpr TI N_EPOCHS = MODE == BENCHMARK_MODE::LARGE ? 5 : (MODE == BENCHMARK_MODE::TINY ? 1 : 2);
        static constexpr T GAMMA = 0.9;
    };
    static constexpr TI BATCH_SIZE = MODE == BENCHMARK_MODE::LARGE ? 512 : (MODE == BENCHMARK_MODE::MEDIUM ? 256 : (MODE == BENCHMARK_MODE::SMALL ? 64 : (MODE == BENCHMARK_MODE::TINY ? 32 : 64)));
    static constexpr TI ACTOR_HIDDEN_DIM = MODE == BENCHMARK_MODE::LARGE ? 256 : (MODE == BENCHMARK_MODE::MEDIUM ? 64 : (MODE == BENCHMARK_MODE::SMALL ? 32 : (MODE == BENCHMARK_MODE::TINY ? 16 : 64)));
    static constexpr TI CRITIC_HIDDEN_DIM = MODE == BENCHMARK_MODE::LARGE ? 256 : (MODE == BENCHMARK_MODE::MEDIUM ? 64 : (MODE == BENCHMARK_MODE::SMALL ? 32 : (MODE == BENCHMARK_MODE::TINY ? 16 : 64)));
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = MODE == BENCHMARK_MODE::MEDIUM ? 1024 : 256;
    static constexpr TI N_ENVIRONMENTS = MODE == BENCHMARK_MODE::LARGE ? 16 : 4;
    static constexpr TI TOTAL_STEP_LIMIT = 300000;
    static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI EPISODE_STEP_LIMIT = 200;
};
template <BENCHMARK_MODE MODE>
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS<MODE>, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsMLP>;
template <typename NEXT>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 1;
    static constexpr TI NUM_EVALUATION_EPISODES = 1000;
    static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};

template <BENCHMARK_MODE MODE>
void run(TI seed, bool verbose){
    DEVICE device;
#ifndef BENCHMARK
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
    ts.actor_optimizer.parameters.alpha = 1e-3;
    ts.critic_optimizer.parameters.alpha = 1e-3;
    while(!rlt::step(device, ts)){
    }
}

void run(TI seed, bool verbose, BENCHMARK_MODE mode){
    if(mode == BENCHMARK_MODE::LARGE){
        run<BENCHMARK_MODE::LARGE>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::MEDIUM){
        run<BENCHMARK_MODE::MEDIUM>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::SMALL){
        run<BENCHMARK_MODE::SMALL>(seed, verbose);
    }else if(mode == BENCHMARK_MODE::TINY){
        run<BENCHMARK_MODE::TINY>(seed, verbose);
    }else{
        std::cout << "Unknown benchmark mode: " << static_cast<TI>(mode) << std::endl;
    }
}


int main(int argc, char** argv) {
    TI seed = 0;
    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }
    BENCHMARK_MODE mode = BENCHMARK_MODE::MEDIUM;
    if (argc > 2) {
        mode = static_cast<BENCHMARK_MODE>(std::atoi(argv[2]));
    }
    bool verbose = true;
    if (argc > 3) {
        verbose = std::atoi(argv[3]);
    }
    run(seed, verbose, mode);
    return 0;
}
