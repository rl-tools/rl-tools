#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include "td3/parameters.h"

#include <thread>

namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

using DEVICE = bpt::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE::index_t;
using penv = parameters_0::environment<T, TI, parameters::DefaultAblationSpec>;
using UI = bpt::rl::environments::multirotor::UI<penv::ENVIRONMENT>;

int main(){
    DEVICE device;
    penv::ENVIRONMENT env;
    UI ui;
    env.parameters = penv::parameters;
    ui.host = "localhost";
    ui.port = "8080";
    bpt::init(device, env, ui);
    penv::ENVIRONMENT::State state, next_state;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, penv::ENVIRONMENT::ACTION_DIM>> action;
    bpt::malloc(device, action);
//    env.parameters.dynamics.rpm_time_constant = 0.01;



    bpt::initial_state(device, env, state);
    bpt::set_all(device, action, 0);
    set(action, 0, 0, 0.5);
    set(action, 0, 1, 0.5);
    bpt::set_state(device, ui, state, action);
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM{}, 0);
    for(TI step_i=0; step_i<10; ++step_i){
        bpt::step(device, env, state, action, next_state, rng);
        state = next_state;
        bpt::set_state(device, ui, state, action);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    bpt::free(device, action);
    return 0;
}