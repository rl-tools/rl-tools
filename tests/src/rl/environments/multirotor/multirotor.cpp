
using DTYPE = double;
using COUNTER_TYPE = int;
#include <cmath>
namespace dynamics_legacy{
    #include "multirotor_dynamics_generic.h"
    #include "parameters.h"
}

constexpr auto STATE_DIM = dynamics_legacy::STATE_DIM;
constexpr auto ACTION_DIM = dynamics_legacy::ACTION_DIM;

#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/multirotor/parameters/default.h>

#include <backprop_tools/rl/environments/multirotor/multirotor.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>

#include <backprop_tools/utils/generic/memcpy.h>

namespace lic = backprop_tools;

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR, MULTIROTOR) {
    using DEVICE = lic::devices::DefaultCPU;


    typename DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;

    const auto parameters = lic::rl::environments::multirotor::parameters::default_parameters<DTYPE, typename DEVICE::index_t>;
    using PARAMETERS = decltype(parameters);
    using REWARD_FUNCTION = PARAMETERS::MDP::REWARD_FUNCTION;
    using SPEC = lic::rl::environments::multirotor::Specification<DTYPE, DEVICE::index_t, PARAMETERS, lic::rl::environments::multirotor::StaticParameters>;
    using ENVIRONMENT = lic::rl::environments::Multirotor<SPEC>;
    std::cout << "sizeof state: " << sizeof(ENVIRONMENT::State) << std::endl;


    ENVIRONMENT env({parameters});
    std::mt19937 rng(0);

    for(COUNTER_TYPE step_i = 0; step_i < 100; step_i++){
        std::normal_distribution<DTYPE> action_distribution;
        DTYPE state[STATE_DIM];
//        for(int i = 0; i < STATE_DIM; i++){
//            state[i] = 0;
//        }
//        state[3] = 1;
        ENVIRONMENT::State env_state;
        lic::sample_initial_state(device, env, env_state, rng);
        for(int i = 0; i < STATE_DIM; i++){
            state[i] = env_state.state[i];
        }

        lic::utils::memcpy(env_state.state, state, STATE_DIM);
        lic::MatrixDynamic<lic::matrix::Specification<DTYPE, typename DEVICE::index_t, 1, ACTION_DIM>> env_action;
        lic::malloc(device, env_action);

        for(COUNTER_TYPE substep_i = 0; substep_i < 100; substep_i++){
            DTYPE action[ACTION_DIM];
            DTYPE dsdt[STATE_DIM];
            DTYPE next_state[STATE_DIM];
            ENVIRONMENT::State env_next_state;
            constexpr DTYPE action_min = 0;
            constexpr DTYPE action_max = 2000;
            for(COUNTER_TYPE action_i = 0; action_i < ACTION_DIM; action_i++){
                action[action_i] = 1000 + lic::math::clamp<DTYPE>(typename DEVICE::SPEC::MATH(), action_distribution(rng) * 500, action_min, action_max);
            }
            for(COUNTER_TYPE action_i = 0; action_i < ACTION_DIM; action_i++){
                set(env_action, 0, action_i, (action[action_i] - action_min) / (action_max - action_min) * 2 - 1);
            }

            // Legacy
            dynamics_legacy::multirotor_dynamics(dynamics_legacy::params, state, action, dsdt);
            dynamics_legacy::next_state_rk4(dynamics_legacy::params, state, action, dynamics_legacy::params.dt, next_state);
            DTYPE quatnorm[4];
            dynamics_legacy::normalize<DTYPE, 4>(&next_state[3], quatnorm);
            for(int i = 0; i < 4; i++){
                next_state[3 + i] = quatnorm[i];
            }



            // Env based
            lic::step(device, env, env_state, env_action, env_next_state);

            DTYPE acc = 0;
            for(COUNTER_TYPE state_i = 0; state_i < STATE_DIM; state_i++){
                acc += std::abs(env_next_state.state[state_i] - next_state[state_i]);
                EXPECT_NEAR(env_next_state.state[state_i], next_state[state_i], 1e-6);
            }
            std::cout << "Next state deviation: " << acc << std::endl;

            for(COUNTER_TYPE state_i = 0; state_i < STATE_DIM; state_i++){
                state[state_i] = next_state[state_i];
            }
            env_state = env_next_state;
        }
        lic::free(device, env_action);

    }

}
