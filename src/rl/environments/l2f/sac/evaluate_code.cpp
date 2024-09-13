#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/layers/dense/operations_cpu.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/sequential_v2/operations_generic.h>
#include <rl_tools/rl/environments/l2f/operations_cpu.h>
#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/extrack/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/loop/steps/checkpoint/config.h>
#include <rl_tools/rl/loop/steps/save_trajectories/config.h>
#include <rl_tools/rl/loop/steps/timing/config.h>


#include <rl_tools/ui_server/client/operations_websocket.h>


#include "../../../experiments/2024-09-13_14-41-33/953f9d3_sequential_algorithm_environment_seq-len/sac_l2f_10/0003/steps/000000001300000/checkpoint.h"

namespace rlt = rl_tools;

#include <regex>
#include <iostream>
#include <filesystem>


#include "approximators.h"


constexpr bool ORIGINAL_CONDITIONS = true;
constexpr bool AUTOMATIC_RESET = false;
constexpr bool ENV_ZERO_ORIENTATION_INIT = false;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

#include "parameters.h"

struct DEFAULT_CONFIG: rl_tools::rl::environments::l2f::parameters::DEFAULT_CONFIG{
    static constexpr bool ZERO_ORIENTATION_INIT = ORIGINAL_CONDITIONS || ZERO_ORIENTATION_INIT;
};

using EVALUATION_ENVIRONMENT = typename env_param_builder::ENVIRONMENT_PARAMETERS<DEFAULT_CONFIG>::ENVIRONMENT;


int main(){
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    rl_tools::checkpoint::actor::MODEL::Buffer<1> buffer;
    rlt::malloc(device, buffer);

    using ENVIRONMENT_UI = rlt::ui_server::client::UIWebSocket<EVALUATION_ENVIRONMENT>;
    ENVIRONMENT_UI ui;

    EVALUATION_ENVIRONMENT env;
    EVALUATION_ENVIRONMENT::Parameters parameters;
    EVALUATION_ENVIRONMENT::State state, next_state;


    rlt::malloc(device, env);
    rlt::sample_initial_parameters(device, env, parameters, rng);

    using STEP_BY_STEP_MODE = rlt::nn::layers::gru::StepByStepMode<rlt::mode::Default<>, rlt::nn::layers::gru::StepByStepModeSpecification<TI, ORIGINAL_CONDITIONS || AUTOMATIC_RESET>>;
    rlt::Mode<STEP_BY_STEP_MODE> mode;

    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, EVALUATION_ENVIRONMENT::Observation::DIM>>> observation;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, 1, 1, EVALUATION_ENVIRONMENT::ACTION_DIM>>> action;
    rlt::malloc(device, observation);
    rlt::malloc(device, action);

    auto observation_matrix_view = rlt::matrix_view(device, observation);
    auto action_matrix_view = rlt::matrix_view(device, action);




    rlt::init(device, env, parameters, ui);
    TI step = 0;
    mode.reset = true;
    bool truncated = true;
    T cumulative_rewards = 0;
    while(true){
        if(truncated){
            rlt::sample_initial_state(device, env, parameters, state, rng);
        }
        mode.step = step;

        rlt::observe(device, env, parameters, state, EVALUATION_ENVIRONMENT::Observation{}, observation_matrix_view, rng);
        rlt::evaluate(device, rl_tools::checkpoint::actor::module, observation, action, buffer, rng, mode);
        T dt = rlt::step(device, env, parameters, state, action_matrix_view, next_state, rng);
        state = next_state;
        T reward = rlt::reward(device, env, parameters, state, action_matrix_view, next_state, rng);
        cumulative_rewards += reward;

        rlt::set_state(device, env, parameters, ui, state);
        std::this_thread::sleep_for(std::chrono::milliseconds((TI)(1000 * dt)));
        bool terminated = rlt::terminated(device, env, parameters, state, rng);
        truncated = terminated || step >= 500;
        mode.reset = truncated;
        if(truncated){
            std::cout << "Episode terminated after " << step << " steps with cumulative rewards: " << cumulative_rewards << std::endl;
            step = 0;
            cumulative_rewards = 0;
        }
        step++;
    }




    return 0;
}
