#include <gtest/gtest.h>

#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/environments/pendulum/ui.h>
namespace bpt = backprop_tools;
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_PENDULUM_RENCER, RENDER) {
    typedef double T;
    typedef bpt::rl::environments::pendulum::Spec<T, bpt::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
    typedef bpt::rl::environments::Pendulum::CPU<PENDULUM_SPEC> ENVIRONMENT;
    for(int j=0; j < 10; j++){
        bpt::rl::environments::pendulum::UI<T> ui_state;
        for(int i = 0; i < 100; i++){
            ui_state.angle = (T)i/100 * 2 * M_PI;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

