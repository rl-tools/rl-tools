#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/acrobot/operations_cpu.h>
#include <rl_tools/ui_server/client/operations_websocket.h>


namespace rlt = rl_tools;


using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using T = float;

using ENVIRONMENT_PARAMETERS = rlt::rl::environments::acrobot::DefaultParameters<T>;
using ENVIRONMENT_SPEC = rlt::rl::environments::acrobot::Specification<T, TI, ENVIRONMENT_PARAMETERS>;
using ENVIRONMENT = rlt::rl::environments::AcrobotSwingup<ENVIRONMENT_SPEC>;

using ENV_UI = rlt::ui_server::client::UIWebSocket<ENVIRONMENT>;

int main(){
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);
    ENVIRONMENT env;
    ENV_UI ui;
    rlt::init(device, env, ui);
    typename ENVIRONMENT::State state;
    rlt::sample_initial_state(device, env, state, rng);
    while(true){
        rlt::set_state(device, env, ui, state);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    return 0;
}
