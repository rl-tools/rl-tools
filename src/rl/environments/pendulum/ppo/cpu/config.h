//#define RL_TOOLS_BACKEND_DISABLE_BLAS
#ifdef RL_TOOLS_STATIC_MEM
#define RL_TOOLS_DISABLE_DYNAMIC_MEMORY_ALLOCATIONS
#endif

#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/rl/environment_wrappers/scale_observations/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
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


template <typename DEVICE, typename TYPE_POLICY, bool DYNAMIC_ALLOCATION>
struct CONFIG_FACTORY{
    using TI = typename DEVICE::index_t;
    using T = typename TYPE_POLICY::DEFAULT;
    using RNG = typename DEVICE::SPEC::RANDOM::template ENGINE<>;
    using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<typename TYPE_POLICY::DEFAULT, TI, rlt::rl::environments::pendulum::DefaultParameters<typename TYPE_POLICY::DEFAULT>>;
    using PRE_ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
    using SCALE_OBSERVATIONS_WRAPPER_SPEC = rlt::rl::environment_wrappers::scale_observations::Specification<TYPE_POLICY, TI>;
    using ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_OBSERVATIONS_WRAPPER_SPEC, PRE_ENVIRONMENT>;

    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::ppo::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
        static constexpr TI BATCH_SIZE = 256;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
        static constexpr TI ON_POLICY_RUNNER_STEPS_PER_ENV = 1024;
        static constexpr TI N_ENVIRONMENTS = 4;
        static constexpr TI TOTAL_STEP_LIMIT = 300000;
        static constexpr TI STEP_LIMIT = TOTAL_STEP_LIMIT/(ON_POLICY_RUNNER_STEPS_PER_ENV * N_ENVIRONMENTS) + 1;
        static constexpr TI EPISODE_STEP_LIMIT = 200;
        using OPTIMIZER_PARAMETERS = rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_PYTORCH<TYPE_POLICY>;
        struct PPO_PARAMETERS: rlt::rl::algorithms::ppo::DefaultParameters<TYPE_POLICY, TI, BATCH_SIZE>{
            static constexpr T ACTION_ENTROPY_COEFFICIENT = 0.0;
            static constexpr TI N_EPOCHS = 2;
            static constexpr T GAMMA = 0.9;
            static constexpr T INITIAL_ACTION_STD = 2.0;
            static constexpr bool NORMALIZE_OBSERVATIONS = true;
        };
    };
    using LOOP_CORE_CONFIG = rlt::rl::algorithms::ppo::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::ppo::loop::core::ConfigApproximatorsSequential, DYNAMIC_ALLOCATION>;
    template <typename NEXT>
    struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<TYPE_POLICY, TI, NEXT>{
        static constexpr TI EVALUATION_INTERVAL = 10;
        static constexpr TI NUM_EVALUATION_EPISODES = 100;
        static constexpr TI N_EVALUATIONS = NEXT::CORE_PARAMETERS::STEP_LIMIT / EVALUATION_INTERVAL;
    };

    #ifndef BENCHMARK
    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS<LOOP_CORE_CONFIG>>;
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_EVAL_CONFIG>;
    #else
    using LOOP_TIMING_CONFIG = rlt::rl::loop::steps::timing::Config<LOOP_CORE_CONFIG>;
    #endif

    using LOOP_CONFIG = LOOP_TIMING_CONFIG;
    using LOOP_STATE = typename LOOP_CONFIG::template State<LOOP_CONFIG>;

    static constexpr TI NUM_EPISODES_FINAL_EVAL = 1000;
    using EVAL_SPEC = rlt::rl::utils::evaluation::Specification<TYPE_POLICY, TI, typename LOOP_CONFIG::ENVIRONMENT_EVALUATION, NUM_EPISODES_FINAL_EVAL, ENVIRONMENT::EPISODE_STEP_LIMIT>;

    using POLICY = rlt::utils::typing::remove_reference_t<decltype(rlt::get_actor(LOOP_STATE{}))>;
    using EVAL_BUFFER = rlt::rl::utils::evaluation::PolicyBuffer<rlt::rl::utils::evaluation::PolicyBufferSpecification<EVAL_SPEC, POLICY, DYNAMIC_ALLOCATION>>;
};
