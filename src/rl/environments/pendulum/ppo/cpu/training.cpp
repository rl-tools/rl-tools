#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp_unconditional_stddev/operations_generic.h>


#include <rl_tools/rl/algorithms/ppo/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>
#include <rl_tools/rl/algorithms/ppo/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>
#include <rl_tools/rl/loop/steps/timing/operations.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<T, TI>{
    static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
    static constexpr T INITIAL_ACTION_STD = 1.0;
};
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    using PPO_PARAMETERS = ::PPO_PARAMETERS;
    static constexpr TI ACTOR_NUM_LAYERS = 3;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr TI CRITIC_HIDDEN_DIM = 64;
    static constexpr TI N_ENVIRONMENTS = 4;
    static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024/N_ENVIRONMENTS;
    static constexpr TI BATCH_SIZE = 256;
    static constexpr TI STEP_LIMIT = 100000/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
    static constexpr TI ON_POLICY_RUNNER_STEP_LIMIT = 200;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::FAST_TANH;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::DefaultConfigApproximatorsMLP>;
template <typename NEXT=LOOP_CORE_CONFIG>
struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::DefaultParameters<T, TI, NEXT>{
    static constexpr TI EVALUATION_INTERVAL = 10;
    static constexpr TI NUM_EVALUATION_EPISODES = 100;
    static constexpr TI N_EVALUATIONS = NEXT::PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
};
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<>>;
using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::DefaultConfig<LOOP_EVAL_CONFIG>;
using LOOP_CONFIG = LOOP_TIMING_CONFIG;
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main(int argc, char** argv) {
    TI seed = 0;
    if (argc > 1) {
        seed = std::atoi(argv[1]);
    }
    LOOP_STATE ts;
    rlt::init(ts, seed);
    ts.actor_optimizer.parameters.alpha = 1e-3;
    ts.critic_optimizer.parameters.alpha = 1e-3;
    while(!rlt::step(ts)){
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
    }
    return 0;
}
