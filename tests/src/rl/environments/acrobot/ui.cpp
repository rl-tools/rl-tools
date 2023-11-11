#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/acrobot/operations_generic.h>
#include <rl_tools/rl/environments/acrobot/ui.h>
namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include "../../../utils/utils.h"
#include <gtest/gtest.h>

TEST(RL_TOOLS_RL_ENVIRONMENTS_ACROBOT_TEST, UI) {
    using T = float;
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using ENVIRONMENT_SPEC = rlt::rl::environments::acrobot::Specification<T, TI, rlt::rl::environments::acrobot::DefaultParameters<T>>;
    using ENVIRONMENT = rlt::rl::environments::Acrobot<ENVIRONMENT_SPEC>;
    using UI_SPEC = rlt::rl::environments::acrobot::ui::Specification<T, TI, ENVIRONMENT, 500, 60>;
    using UI = rlt::rl::environments::acrobot::UI<UI_SPEC>;

    DEVICE device;
    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);
    ENVIRONMENT env;
    ENVIRONMENT::State state, next_state;
    UI ui;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, 1>> action;
    rlt::malloc(device, action);

    rlt::init(device, env, ui);

    for(TI episode_i = 0; episode_i < 10; episode_i++){
        rlt::sample_initial_state(device, env, state, rng);
        for(TI step_i = 0; step_i < 10; step_i++){
            rlt::step(device, env, state, action, next_state, rng);
            state = next_state;
            rlt::set_state(device, env, ui, state);
            rlt::render(device, env, ui);
        }
    }

}
