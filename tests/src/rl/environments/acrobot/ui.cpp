#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/acrobot/operations_generic.h>
#include <backprop_tools/rl/environments/acrobot/ui.h>
namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

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
    ENVIRONMENT::State state, next_state;
    UI ui;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, 1>> action;
    bpt::malloc(device, action);

    bpt::init(device, env, ui);

    for(TI episode_i = 0; episode_i < 10; episode_i++){
        bpt::sample_initial_state(device, env, state, rng);
        for(TI step_i = 0; step_i < 10; step_i++){
            bpt::step(device, env, state, action, next_state, rng);
            state = next_state;
            bpt::set_state(device, env, ui, state);
            bpt::render(device, env, ui);
        }
    }

}
