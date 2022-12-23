#include <gtest/gtest.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/pendulum/ui.h>
namespace lic = layer_in_c;
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_RENCER, RENDER) {
    typedef double T;
    typedef lic::rl::environments::pendulum::Spec<T, lic::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
    typedef lic::rl::environments::Pendulum<lic::devices::Generic, PENDULUM_SPEC> ENVIRONMENT;
    for(int j=0; j < 10; j++){
        lic::rl::environments::pendulum::UI<T> ui_state;
        for(int i = 0; i < 100; i++){
            ui_state.angle = (T)i/100 * 2 * M_PI;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

