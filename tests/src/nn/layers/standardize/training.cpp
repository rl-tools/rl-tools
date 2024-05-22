#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/standardize/operations_generic.h>
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

namespace config{
    using namespace rlt;
    template <typename T_ENVIRONMENT>
    struct _LoopConfig{
        struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, T_ENVIRONMENT>{
            static constexpr TI STEP_LIMIT = 20000;
            static constexpr TI ACTOR_NUM_LAYERS = 3;
            static constexpr TI ACTOR_HIDDEN_DIM = 64;
            static constexpr TI CRITIC_NUM_LAYERS = 3;
            static constexpr TI CRITIC_HIDDEN_DIM = 64;
        };
        template<typename T, typename TI, typename ENVIRONMENT, typename PARAMETERS, typename CONTAINER_TYPE_TAG>
        struct ConfigApproximatorsSequential{
            template <typename CAPABILITY>
            struct ACTOR{
                static constexpr TI HIDDEN_DIM = PARAMETERS::ACTOR_HIDDEN_DIM;
                static constexpr TI BATCH_SIZE = PARAMETERS::SAC_PARAMETERS::ACTOR_BATCH_SIZE;
                static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::ACTOR_ACTIVATION_FUNCTION;
                using LAYER_0_SPEC = nn::layers::standardize::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, BATCH_SIZE, CONTAINER_TYPE_TAG>;
                using LAYER_0 = nn::layers::standardize::BindSpecification<LAYER_0_SPEC>;
                using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, BATCH_SIZE, nn::parameters::groups::Normal, CONTAINER_TYPE_TAG>;
                using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
                using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, BATCH_SIZE, nn::parameters::groups::Normal, CONTAINER_TYPE_TAG>;
                using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
                static constexpr TI ACTOR_OUTPUT_DIM = ENVIRONMENT::ACTION_DIM * 2; // to express mean and log_std for each action
                using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, ACTOR_OUTPUT_DIM, nn::activation_functions::ActivationFunction::IDENTITY, BATCH_SIZE, nn::parameters::groups::Output, CONTAINER_TYPE_TAG>; // note the output activation should be identity because we want to sample from a gaussian and then squash afterwards (taking into account the squashing in the distribution)
                using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

                using IF = nn_models::sequential::Interface<CAPABILITY>;
                using MODEL = typename IF::template Module<LAYER_0::template Layer, typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>>;
            };

            template <typename CAPABILITY>
            struct CRITIC{
                static constexpr TI HIDDEN_DIM = PARAMETERS::CRITIC_HIDDEN_DIM;
                static constexpr TI BATCH_SIZE = PARAMETERS::SAC_PARAMETERS::CRITIC_BATCH_SIZE;
                static constexpr auto ACTIVATION_FUNCTION = PARAMETERS::CRITIC_ACTIVATION_FUNCTION;

                using LAYER_1_SPEC = nn::layers::dense::Specification<T, TI, ENVIRONMENT::OBSERVATION_DIM + ENVIRONMENT::ACTION_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, BATCH_SIZE, nn::parameters::groups::Input, CONTAINER_TYPE_TAG>;
                using LAYER_1 = nn::layers::dense::BindSpecification<LAYER_1_SPEC>;
                using LAYER_2_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, HIDDEN_DIM, ACTIVATION_FUNCTION, BATCH_SIZE, nn::parameters::groups::Normal, CONTAINER_TYPE_TAG>;
                using LAYER_2 = nn::layers::dense::BindSpecification<LAYER_2_SPEC>;
                using LAYER_3_SPEC = nn::layers::dense::Specification<T, TI, HIDDEN_DIM, 1, nn::activation_functions::ActivationFunction::IDENTITY, BATCH_SIZE, nn::parameters::groups::Output, CONTAINER_TYPE_TAG>;
                using LAYER_3 = nn::layers::dense::BindSpecification<LAYER_3_SPEC>;

                using IF = nn_models::sequential::Interface<CAPABILITY>;
                using MODEL = typename IF::template Module<LAYER_1::template Layer, typename IF::template Module<LAYER_2::template Layer, typename IF::template Module<LAYER_3::template Layer>>>;
            };

            using OPTIMIZER_SPEC = nn::optimizers::adam::Specification<T, TI, typename PARAMETERS::OPTIMIZER_PARAMETERS>;

            using OPTIMIZER = nn::optimizers::Adam<OPTIMIZER_SPEC>;

            using ACTOR_TYPE = typename ACTOR<nn::layer_capability::Gradient<nn::parameters::Adam>>::MODEL;
            using CRITIC_TYPE = typename CRITIC<nn::layer_capability::Gradient<nn::parameters::Adam>>::MODEL;
            using CRITIC_TARGET_TYPE = typename CRITIC<nn::layer_capability::Forward>::MODEL;
        };
        using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<T, TI, RNG, T_ENVIRONMENT, LOOP_CORE_PARAMETERS, ConfigApproximatorsSequential>;
        struct LOOP_EVAL_PARAMETERS: rlt::rl::loop::steps::evaluation::Parameters<T, TI, LOOP_CORE_CONFIG>{
            static constexpr TI NUM_EVALUATION_EPISODES = 100;
        };
        using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::Config<LOOP_CORE_CONFIG, LOOP_EVAL_PARAMETERS>;
        using LOOP_CONFIG = LOOP_EVAL_CONFIG;
    };
}

template <TI T_SCALE>
struct SCALE_OBSERVATIONS_WRAPPER_SPEC: rlt::rl::environment_wrappers::scale_observations::Specification<T, TI>{
    static constexpr T SCALE = T_SCALE;
};

template <TI SCALE, TI N_SEEDS=10>
T lifetime_return(){
    using WRAPPED_ENVIRONMENT = rlt::rl::environment_wrappers::ScaleObservations<SCALE_OBSERVATIONS_WRAPPER_SPEC<SCALE>, ENVIRONMENT>;
    using LOOP_CONFIG = typename config::_LoopConfig<WRAPPED_ENVIRONMENT>::LOOP_CONFIG;
    DEVICE device;
    typename LOOP_CONFIG::template State<LOOP_CONFIG> ts;

    T acc = 0;
#ifdef RL_TOOLS_TESTS_CODE_COVERAGE
    static constexpr TI N_SEEDS_ACTUAL = 1;
#else
    static constexpr TI N_SEEDS_ACTUAL = N_SEEDS;
#endif
    for(TI seed = 0; seed < N_SEEDS; seed++){
        rlt::malloc(device);
        rlt::init(device);
        rlt::malloc(device, ts);
        rlt::init(device, ts, seed);
        while(!rlt::step(device, ts)){
#ifdef RL_TOOLS_TESTS_CODE_COVERAGE
            if(ts.step > 2000){
                break;
            }
#endif
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
#ifndef RL_TOOLS_TESTS_CODE_COVERAGE
    ASSERT_GT(return_10000x / return_10x, 3);
#endif
}
