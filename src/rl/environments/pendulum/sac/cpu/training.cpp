#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/operations_cpu_mux.h>
//#include <rl_tools/nn_models/output_view/model.h>
#include <rl_tools/rl/environments/pendulum/operations_cpu.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>


#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/evaluation/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations.h>
#include <rl_tools/rl/algorithms/sac/loop/evaluation/operations.h>

namespace rlt = rl_tools;

//#include "training.h"
using DEVICE = rlt::devices::DefaultCPU;
using T = float;
using TI = typename DEVICE::index_t;

using PENDULUM_SPEC = rlt::rl::environments::pendulum::Specification<T, TI, rlt::rl::environments::pendulum::DefaultParameters<T>>;
using ENVIRONMENT = rlt::rl::environments::Pendulum<PENDULUM_SPEC>;
using LOOP_CORE_CONFIG = rlt::rl::algorithms::sac::loop::core::DefaultConfig<DEVICE, T, ENVIRONMENT>;
struct LOOP_CORE_CONFIG_CUSTOM: LOOP_CORE_CONFIG{
//    static constexpr TI STEP_LIMIT = 10000;
};
using LOOP_EVAL_CONFIG = rlt::rl::algorithms::sac::loop::evaluation::DefaultConfig<LOOP_CORE_CONFIG_CUSTOM>;
using LOOP_CONFIG = LOOP_EVAL_CONFIG;

using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

int main(){

    LOOP_STATE ts;

    rlt::init(ts, 0);

    while(!rlt::step(ts)){ }
    rlt::destroy(ts);
}