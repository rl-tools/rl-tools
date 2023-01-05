
using DTYPE = double;
using COUNTER_TYPE = int;
namespace dynamics_legacy{
    #include "multirotor_dynamics_generic.h"
    #include "parameters.h"
}

constexpr auto STATE_DIM = dynamics_legacy::STATE_DIM;
constexpr auto ACTION_DIM = dynamics_legacy::ACTION_DIM;

#include <layer_in_c/utils/generic/math.h>
namespace lic = layer_in_c;

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
TEST(LAYER_IN_C_RL_ENVIRONMENTS_MULTIROTOR, MULTIROTOR) {
    std::mt19937 rng(0);

    for(COUNTER_TYPE step_i = 0; step_i < 100; step_i++){
        std::normal_distribution<DTYPE> action_distribution;
        DTYPE state[STATE_DIM];
        for(COUNTER_TYPE substep_i = 0; substep_i < 10; substep_i++){
            DTYPE action[ACTION_DIM];
            DTYPE dsdt[STATE_DIM];
            DTYPE next_state[STATE_DIM];
            for(COUNTER_TYPE action_i = 0; action_i < ACTION_DIM; action_i++){
                action[action_i] = 1000 + lic::utils::math::clamp<DTYPE>(action_distribution(rng) * 500, 0, 2000);
            }
            dynamics_legacy::multirotor_dynamics(dynamics_legacy::params, state, action, dsdt);
            dynamics_legacy::next_state_rk4(dynamics_legacy::params, state, action, dynamics_legacy::params.dt, next_state);
            for(COUNTER_TYPE state_i = 0; state_i < STATE_DIM; state_i++){
                state[state_i] = next_state[state_i];
            }
        }

    }

}
