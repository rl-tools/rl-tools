#include <gtest/gtest.h>

#include <layer_in_c/rl/environments/environments.h>
#include <layer_in_c/rl/environments/pendulum/ui.h>
namespace lic = layer_in_c;
#define DTYPE float
const DTYPE STATE_TOLERANCE = 0.00001;

TEST(LAYER_IN_C_RL_ENVIRONMENTS_PENDULUM_RENCER, RENDER) {
    typedef double T;
    typedef lic::rl::environments::pendulum::Spec<T, lic::rl::environments::pendulum::DefaultParameters<T>> PENDULUM_SPEC;
    typedef lic::rl::environments::pendulum::Pendulum<lic::devices::Generic, PENDULUM_SPEC> ENVIRONMENT;
    ENVIRONMENT env;
    T state[ENVIRONMENT::STATE_DIM];
//    std::mt19937 rng;
//    sample_initial_state(pendulum, state, rng);
    T initial_state[ENVIRONMENT::STATE_DIM] = {0.58335993034834344, 0.68853148851319657};
    memcpy(state, initial_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
    T next_state[ENVIRONMENT::STATE_DIM];
    T r = 0;
    for(int i = 0; i < 5; i++){
        T action[ENVIRONMENT::ACTION_DIM] = {-1};
        r += lic::step(env, state, action, next_state);
        memcpy(state, next_state, sizeof(T) * ENVIRONMENT::STATE_DIM);
        std::cout << "state: " << state[0] << ", " << state[1] << std::endl;
    }
    for(int i = 0; i < 100; i++){
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        lic::rl::environments::pendulum::UI ui_state;
    }
}

