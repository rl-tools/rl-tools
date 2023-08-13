#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/acrobot/operations_generic.h>
#include <backprop_tools/rl/environments/acrobot/ui.h>
namespace bpt = backprop_tools;

#include "../../../utils/utils.h"
#include <gtest/gtest.h>

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_ACROBOT_TEST, UI) {
    using T = float;
    using DEVICE = bpt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using ENVIRONMENT_SPEC = bpt::rl::environments::acrobot::Specification<T, TI, bpt::rl::environments::acrobot::DefaultParameters<T>>;
    using ENVIRONMENT = bpt::rl::environments::Acrobot<ENVIRONMENT_SPEC>;
    using UI_SPEC = bpt::rl::environments::acrobot::ui::Specification<T, TI, ENVIRONMENT, 500, 60>;
    using UI = bpt::rl::environments::acrobot::UI<UI_SPEC>;

    DEVICE device;
    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    ENVIRONMENT env;
    ENVIRONMENT::State state;
    UI ui;

    bpt::init(device, env, ui);

    bpt::sample_initial_state(device, env, state, rng);

    while(true){
        state.theta_0 += 0.1;
        state.theta_1 += 0.1;
        bpt::set_state(device, env, ui, state);
        bpt::render(device, env, ui);
    }

}
