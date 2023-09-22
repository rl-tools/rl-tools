
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

namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

#include <gtest/gtest.h>
#include <random>
#include <stdint.h>
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR, MULTIROTOR) {
    using DEVICE = bpt::devices::DefaultCPU;
    using TI = typename DEVICE::index_t;


    DEVICE device;

    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM{}, 2);

    const auto parameters = bpt::rl::environments::multirotor::parameters::default_parameters<DTYPE, typename DEVICE::index_t>;
    using PARAMETERS = decltype(parameters);
    using REWARD_FUNCTION = PARAMETERS::MDP::REWARD_FUNCTION;
    using SPEC = bpt::rl::environments::multirotor::Specification<DTYPE, DEVICE::index_t, PARAMETERS, bpt::rl::environments::multirotor::StaticParametersDefault<DTYPE, TI>>;
    using ENVIRONMENT = bpt::rl::environments::Multirotor<SPEC>;
    std::cout << "sizeof state: " << sizeof(ENVIRONMENT::State) << std::endl;


    ENVIRONMENT env({parameters});

    for(COUNTER_TYPE step_i = 0; step_i < 1000; step_i++){
        DTYPE state[STATE_DIM];
//        for(int i = 0; i < STATE_DIM; i++){
//            state[i] = 0;
//        }
//        state[3] = 1;
        ENVIRONMENT::State env_state;
        bpt::sample_initial_state(device, env, env_state, rng);
        for(int i = 0; i < 3; i++){
            state[i] = env_state.position[i];
        }
        for(int i = 0; i < 4; i++){
            state[3+i] = env_state.orientation[i];
        }
        for(int i = 0; i < 3; i++){
            state[3+4+i] = env_state.linear_velocity[i];
        }
        for(int i = 0; i < 3; i++){
            state[3+4+3+i] = env_state.angular_velocity[i];
        }

//        bpt::utils::memcpy(env_state.state, state, STATE_DIM);
        bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, typename DEVICE::index_t, 1, ACTION_DIM>> env_action;
        bpt::malloc(device, env_action);


        for(COUNTER_TYPE substep_i = 0; substep_i < 100; substep_i++){
            DTYPE dsdt[STATE_DIM];
            DTYPE next_state[STATE_DIM];
            DTYPE action[ACTION_DIM];
            constexpr DTYPE action_min = 0;
            constexpr DTYPE action_max = 2000;
            ENVIRONMENT::State env_next_state;
            DTYPE standard_normal = bpt::random::normal_distribution::sample(DEVICE::SPEC::RANDOM{}, (DTYPE)0, (DTYPE)1, rng);
            for(COUNTER_TYPE action_i = 0; action_i < ACTION_DIM; action_i++){
                action[action_i] = 1000 + bpt::math::clamp<DTYPE>(typename DEVICE::SPEC::MATH(), standard_normal * 500, 0, 1000);
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
            bpt::step(device, env, env_state, env_action, env_next_state, rng);

            DTYPE acc = 0;
            constexpr DTYPE threshold = 1e-10;
            for(COUNTER_TYPE state_i = 0; state_i < 3; state_i++){
                acc += std::abs(env_next_state.position[state_i] - next_state[state_i]);
                if(acc > threshold){
                    std::cout << "break" << std::endl;
                }
                ASSERT_NEAR(env_next_state.position[state_i], next_state[state_i], threshold);
            }
            for(COUNTER_TYPE state_i = 0; state_i < 4; state_i++){
                acc += std::abs(env_next_state.orientation[state_i] - next_state[3+state_i]);
                if(acc > threshold){
                    std::cout << "break" << std::endl;
                }
                ASSERT_NEAR(env_next_state.orientation[state_i], next_state[3+state_i], threshold);
            }
            for(COUNTER_TYPE state_i = 0; state_i < 3; state_i++){
                acc += std::abs(env_next_state.linear_velocity[state_i] - next_state[3+4+state_i]);
                if(acc > threshold){
                    std::cout << "break" << std::endl;
                }
                ASSERT_NEAR(env_next_state.linear_velocity[state_i], next_state[3+4+state_i], threshold);
            }
            for(COUNTER_TYPE state_i = 0; state_i < 3; state_i++){
                acc += std::abs(env_next_state.angular_velocity[state_i] - next_state[3+4+3+state_i]);
                if(acc > threshold){
                    std::cout << "break" << std::endl;
                }
                ASSERT_NEAR(env_next_state.angular_velocity[state_i], next_state[3+4+3+state_i], threshold);
            }
            std::cout << "(Sub)Step: " << substep_i << " Next state deviation: " << acc << std::endl;

            for(COUNTER_TYPE state_i = 0; state_i < STATE_DIM; state_i++){
                state[state_i] = next_state[state_i];
            }
            env_state = env_next_state;
        }
        bpt::free(device, env_action);

    }

}
