#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/car/operations_cpu.h>
#include <rl_tools/rl/environments/car/operations_json.h>

#include <rl_tools/rl/environments/car/operations_cpu.h>
#include <rl_tools/ui_server/client/operations_cpu.h>
namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using RNG = decltype(rlt::random::default_engine(typename DEVICE::SPEC::RANDOM{}));
using T = float;
using TI = typename DEVICE::index_t;

using ENV_SPEC = rlt::rl::environments::car::SpecificationTrack<T, TI, 100, 100, 20>;
using ENVIRONMENT = rlt::rl::environments::CarTrack<ENV_SPEC>;

using UI = rlt::ui_server::client::UI<ENVIRONMENT>;


#include <chrono>
#include <thread>

int main(){
    DEVICE device;
    RNG rng;
    ENVIRONMENT env;
    UI ui;

    rlt::malloc(device, env);
    rlt::init(device, env);
//    rlt::malloc(device, ui);
    rlt::init(device, env, ui);

    std::this_thread::sleep_for(std::chrono::seconds(1000));

    return 0;
}