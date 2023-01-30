#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/nn_models/operations_cpu.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/multirotor/ui.h>
#include <layer_in_c/rl/environments/multirotor/multirotor.h>
#include <layer_in_c/rl/environments/multirotor/parameters/default.h>

#include <layer_in_c/rl/environments/multirotor/operations_cpu.h>

#include <layer_in_c/rl/utils/evaluation.h>
#include <layer_in_c/nn_models/persist.h>

namespace lic = layer_in_c;

using DTYPE = float;
#include "../multirotor_training/parameters.h"

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <highfive/H5File.hpp>

using DEVICE = lic::devices::DefaultCPU;

namespace parameter_set = parameters_0;

using parameters_environment = parameter_set::environment<DEVICE, DTYPE>;
using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;

using parameters_rl = parameter_set::rl<DEVICE, DTYPE, ENVIRONMENT>;


//TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_UI, TEST_UI) {
//    DEVICE::SPEC::LOGGING logger;
//    DEVICE device(logger);
//    lic::rl::environments::multirotor::UI<ENVIRONMENT> ui;
//    ui.host = "localhost";
//    ui.port = "8080";
////    parameters.mdp.init = lic::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
//    auto parameters = parameters_environment::parameters;
//    ENVIRONMENT env({parameters});
//    ENVIRONMENT::State state, next_state;
//    std::mt19937 rng(0);
//    lic::init(device, env, ui);
//    lic::sample_initial_state(device, env, state, rng);
//    for(int i = 0; i < 100; i++){
//        DTYPE action[4];
//        action[0] = 1.0;
//        action[1] = 0.0;
//        action[2] = 0.0;
//        action[3] = 0.0;
//        lic::step(device, env, state, action, next_state);
//        state = next_state;
//        lic::set_state(device, ui, state);
////        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * parameters.integration.dt)));
//        std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
//    }
//}

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_UI, LOAD_ACTOR) {
    DEVICE::SPEC::LOGGING logger;
    DEVICE device(logger);
    lic::rl::environments::multirotor::UI<ENVIRONMENT> ui;
    ui.host = "localhost";
    ui.port = "8080";
//    parameters.mdp.init = lic::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
    auto parameters = parameters_environment::parameters;
    ENVIRONMENT env({parameters});
    ENVIRONMENT::State state, next_state;
    std::mt19937 rng(0);
    lic::init(device, env, ui);

    parameters_rl::ACTOR_NETWORK_TYPE actor;
    lic::malloc(device, actor);

    std::string actor_output_path = "actor.h5";
    auto actor_file = HighFive::File(actor_output_path, HighFive::File::ReadOnly);
    lic::load(device, actor, actor_file.getGroup("actor"));



    for(DEVICE::index_t episode_i = 0; episode_i < 10; episode_i++){
        lic::sample_initial_state(device, env, state, rng);
        lic::rl::utils::evaluation::State<DTYPE, typename ENVIRONMENT::State> eval_state;
        eval_state.state = state;
        for (DEVICE::index_t i = 0; i < 100; i++) {
            if(lic::evaluate_step(device, env, ui, actor, eval_state)){
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds((int)(parameters.integration.dt * 5000)));
        }
//        DTYPE r = lic::evaluate<DEVICE, ENVIRONMENT, decltype(ui), decltype(actor), decltype(rng), parameters_rl::ENVIRONMENT_STEP_LIMIT, true>(device, env, ui, actor, 1, rng);
        std::cout << "return: " << eval_state.episode_return << std::endl;
    }
    lic::free(device, actor);
}
