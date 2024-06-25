#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/multi_agent/bottleneck/operations_cpu.h>
#include <rl_tools/ui_server/client/operations_websocket.h>

namespace rlt = rl_tools;


using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using T = float;

struct ENVIRONMENT_PARAMETERS: rlt::rl::environments::multi_agent::bottleneck::DefaultParameters<T, TI>{
//    static constexpr T DT = 0.018;
};
using ENVIRONMENT_SPEC = rlt::rl::environments::multi_agent::bottleneck::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
using ENVIRONMENT = rlt::rl::environments::multi_agent::Bottleneck<ENVIRONMENT_SPEC>;

using ENV_UI = rlt::ui_server::client::UIWebSocket<ENVIRONMENT>;

int main(){
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    ENVIRONMENT env;
    typename ENVIRONMENT::Parameters env_parameters;
    ENV_UI ui;
    ui.address = "127.0.0.1";
    ui.port = 13337;
    rlt::init(device, env, env_parameters, ui);
    typename ENVIRONMENT::State state, next_state;
    ENVIRONMENT::Parameters parameters;
    rlt::sample_initial_parameters(device, env, parameters, rng);
    rlt::sample_initial_state(device, env, parameters, state, rng);
    rlt::MatrixStatic<rlt::matrix::Specification<T, TI, ENVIRONMENT_PARAMETERS::N_AGENTS, ENVIRONMENT::ACTION_DIM>> action;
//    rlt::randn(device, action, rng);
    rlt::set_all(device, action, 1);
    rlt::clamp(device, action, -1, 1);
    while(true){
        rlt::set_state(device, env, parameters, ui, state, action);
        std::this_thread::sleep_for(std::chrono::duration<T>(0.001));
        T dt = rlt::step(device, env, parameters, state, action, next_state, rng);
        std::this_thread::sleep_for(std::chrono::duration<T>(dt));
        state = next_state;
    }
    return 0;
}
