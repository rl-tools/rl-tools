#include <rl_tools/operations/cpu_mux.h>

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/instance/persist_code.h>

#include <rl_tools/rl/environments/pendulum/operations_generic.h>
#include <rl_tools/rl/environment_wrappers/scale_observations/operations_generic.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>
#include <rl_tools/rl/loop/steps/evaluation/operations_generic.h>

#include <gtest/gtest.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;

template <typename ENVIRONMENT>
struct _LoopConfig{
    struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
        static constexpr TI STEP_LIMIT = 20000;
        static constexpr TI ACTOR_NUM_LAYERS = 3;
        static constexpr TI ACTOR_HIDDEN_DIM = 64;
        static constexpr TI CRITIC_NUM_LAYERS = 3;
        static constexpr TI CRITIC_HIDDEN_DIM = 64;
    };
    using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS, rlt::rl::algorithms::sac::loop::core::ConfigApproximatorsSequential>;
    struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CORE_CONFIG>{
        static constexpr TI NUM_EVALUATION_EPISODES = 100;
    };
    using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS>;
    using LOOP_CONFIG = LOOP_EVAL_CONFIG;
};

template <TI T_SCALE>
struct SCALE_OBSERVATIONS_WRAPPER_SPEC: rlt::rl::environment_wrappers::scale_observations::Specification<T, TI>{
    static constexpr T SCALE = T_SCALE;
};

template <TI SCALE, TI N_SEEDS=10>
T lifetime_return(){
    using WRAPPED_ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_OBSERVATIONS_WRAPPER_SPEC<SCALE>, ENVIRONMENT>;
    using LOOP_CONFIG = typename _LoopConfig<WRAPPED_ENVIRONMENT>::LOOP_CONFIG;
    DEVICE device;
    typename LOOP_CONFIG::template State<LOOP_CONFIG> ts;

    T acc = 0;
    for(TI seed = 0; seed < N_SEEDS; seed++){
        rlt::malloc(device);
        rlt::init(device);
        rlt::malloc(device, ts);
        rlt::init(device, ts, seed);
        while(!rlt::step(device, ts)){
            if(ts.step == 5000){
                std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
            }
        }
        for(TI i = 0; i < LOOP_CONFIG::EVALUATION_PARAMETERS::N_EVALUATIONS; i++){
            acc += ts.evaluation_results[i].returns_mean;
        }
        rlt::free(device);
        rlt::free(device, ts);
    }
    return acc / N_SEEDS;
}

TEST(RL_TOOLS_NN_LAYERS_STANDARDIZE, DETRIMENT_TRAINING) {
//    T return_1x = lifetime_return<1>();
    T return_10x = lifetime_return<10>();
//    T return_100x = lifetime_return<100>();
//    T return_1000x = lifetime_return<1000>();
    T return_10000x = lifetime_return<10000>();

//    std::cout << "return_1x: " << return_1x << std::endl;
    std::cout << "return_10x: " << return_10x << std::endl;
//    std::cout << "return_100x: " << return_100x << std::endl;
//    std::cout << "return_1000x: " << return_1000x << std::endl;
    std::cout << "return_10000x: " << return_10000x << std::endl;
    ASSERT_GT(return_10000x / return_10x, 3);
}