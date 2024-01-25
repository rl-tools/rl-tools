#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>


#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/loop/steps/evaluation/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations.h>
#include <rl_tools/rl/loop/steps/evaluation/operations.h>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<T, TI, ENVIRONMENT>{
    static constexpr TI STEP_LIMIT = 11000;
};
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
using LOOP_EVAL_CONFIG = rlt::rl::loop::steps::evaluation::DefaultConfig<LOOP_CORE_CONFIG>;
using LOOP_CONFIG = LOOP_EVAL_CONFIG;

using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main(){
    LOOP_STATE ts;
    rlt::init(ts, 0);
    while(!rlt::step(ts)){
        if(ts.step == 5000){
            std::cout << "steppin yourself > callbacks 'n' hooks: " << ts.step << std::endl;
        }
    }
    rlt::destroy(ts);
}