#include <layer_in_c/operations/cpu.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/multirotor/ui.h>
#include <layer_in_c/rl/environments/multirotor/multirotor.h>
#include <layer_in_c/rl/environments/multirotor/parameters/default.h>

#include <layer_in_c/rl/environments/multirotor/operations_cpu.h>

namespace lic = layer_in_c;

using DTYPE = float;
#include "../multirotor_training/parameters.h"

#include <gtest/gtest.h>
#include <chrono>
#include <thread>

using DEVICE = lic::devices::DefaultCPU;
//auto parameters = lic::rl::environments::multirotor::parameters::default_parameters<DTYPE, DEVICE::index_t>;
auto parameters = parameters_0::parameters<DTYPE, DEVICE::index_t>;
using PARAMETERS = decltype(parameters);
using REWARD_FUNCTION = PARAMETERS::MDP::REWARD_FUNCTION;
typedef lic::rl::environments::multirotor::Specification<DTYPE, DEVICE::index_t, PARAMETERS, lic::rl::environments::multirotor::StaticParameters> ENVIRONMENT_SPEC;
typedef lic::rl::environments::Multirotor<ENVIRONMENT_SPEC> ENVIRONMENT;


TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR_UI, TEST_UI) {
    DEVICE::SPEC::LOGGING logger;
    DEVICE device(logger);
    lic::rl::environments::multirotor::UI<ENVIRONMENT> ui;
    ui.host = "localhost";
    ui.port = "8080";
//    parameters.mdp.init = lic::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
    ENVIRONMENT env({parameters});
    ENVIRONMENT::State state, next_state;
    std::mt19937 rng(0);
    lic::init(device, env, ui);
    lic::sample_initial_state(device, env, state, rng);
    for(int i = 0; i < 100; i++){
        DTYPE action[4];
        action[0] = 1.0;
        action[1] = 0.0;
        action[2] = 0.0;
        action[3] = 0.0;
        lic::step(device, env, state, action, next_state);
        state = next_state;
        lic::set_state(device, ui, state);
//        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * parameters.integration.dt)));
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
    }
}
