#include <backprop_tools/operations/cpu.h>
#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>
#include "parameters_old.h"
#define BACKPROP_TOOLS_NAMESPACE_WRAPPER backprop_tools_new
#define BACKPROP_TOOLS_DISABLE_INCLUDE_GUARDS
#include <backprop_tools_new/operations/cpu.h>
#include <backprop_tools_new/rl/environments/multirotor/operations_cpu.h>
namespace bpt_old = backprop_tools;
namespace bpt_new = backprop_tools_new::backprop_tools;
#include "parameters_new.h"

using DEVICE_OLD = bpt_old::devices::DefaultCPU;
using DEVICE_NEW = bpt_new::devices::DefaultCPU;
using T = double;
using TI = typename DEVICE_NEW::index_t;
using ENVIRONMENT_NEW = typename parameters_sim2real::environment<T, TI>::ENVIRONMENT;
using ENVIRONMENT_OLD = typename parameters_sim2real_old::environment<T, TI>::ENVIRONMENT;

#include <gtest/gtest.h>
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR, REGRESSION_TEST){
    DEVICE_OLD device_old;
    DEVICE_NEW device_new;

    auto rng_new = bpt_new::random::default_engine(typename DEVICE_NEW::SPEC::RANDOM{});
    auto rng_old = bpt_old::random::default_engine(typename DEVICE_OLD::SPEC::RANDOM{});

    for(TI episode_i=0; episode_i < 10; episode_i++){
        ENVIRONMENT_NEW env_new;
        env_new.parameters = parameters_sim2real::environment<T, TI>::parameters;
        ENVIRONMENT_OLD env_old;
        env_old.parameters = parameters_sim2real_old::environment<T, TI>::parameters;

        ENVIRONMENT_NEW::State state_new, next_state_new;
        ENVIRONMENT_OLD::State state_old, next_state_old;

        bpt_old::MatrixDynamic<bpt_old::matrix::Specification<T, TI, 1, ENVIRONMENT_OLD::OBSERVATION_DIM>> observation_old;
        bpt_old::MatrixDynamic<bpt_old::matrix::Specification<T, TI, 1, ENVIRONMENT_OLD::OBSERVATION_DIM_PRIVILEGED>> observation_privileged_old;
        bpt_new::MatrixDynamic<bpt_old::matrix::Specification<T, TI, 1, ENVIRONMENT_NEW::OBSERVATION_DIM>> observation_new;
        bpt_new::MatrixDynamic<bpt_old::matrix::Specification<T, TI, 1, ENVIRONMENT_NEW::OBSERVATION_DIM_PRIVILEGED>> observation_privileged_new;
        bpt_old::MatrixDynamic<bpt_old::matrix::Specification<T, TI, 1, ENVIRONMENT_OLD::ACTION_DIM>> action_old;
        bpt_new::MatrixDynamic<bpt_new::matrix::Specification<T, TI, 1, ENVIRONMENT_NEW::ACTION_DIM>> action_new;
        bpt_old::malloc(device_old, observation_old);
        bpt_old::malloc(device_old, observation_privileged_old);
        bpt_new::malloc(device_new, observation_new);
        bpt_new::malloc(device_new, observation_privileged_new);
        bpt_old::malloc(device_old, action_old);
        bpt_new::malloc(device_new, action_new);
        bpt_new::set(action_new, 0, 0, 1);
        bpt_new::set(action_new, 0, 1, 0);
        bpt_new::set(action_new, 0, 2, 1);
        bpt_new::set(action_new, 0, 3, 0);

        bpt_old::set(action_old, 0, 0, bpt_new::get(action_new, 0, 0));
        bpt_old::set(action_old, 0, 1, bpt_new::get(action_new, 0, 1));
        bpt_old::set(action_old, 0, 2, bpt_new::get(action_new, 0, 2));
        bpt_old::set(action_old, 0, 3, bpt_new::get(action_new, 0, 3));

        bpt_new::sample_initial_state(device_new, env_new, state_new, rng_new);
        state_old.position[0] = state_new.position[0];
        state_old.position[1] = state_new.position[1];
        state_old.position[2] = state_new.position[2];
        state_old.orientation[0] = state_new.orientation[0];
        state_old.orientation[1] = state_new.orientation[1];
        state_old.orientation[2] = state_new.orientation[2];
        state_old.orientation[3] = state_new.orientation[3];
        state_old.linear_velocity[0] = state_new.linear_velocity[0];
        state_old.linear_velocity[1] = state_new.linear_velocity[1];
        state_old.linear_velocity[2] = state_new.linear_velocity[2];
        state_old.angular_velocity[0] = state_new.angular_velocity[0];
        state_old.angular_velocity[1] = state_new.angular_velocity[1];
        state_old.angular_velocity[2] = state_new.angular_velocity[2];
        state_old.rpm[0] = state_new.rpm[0];
        state_old.rpm[1] = state_new.rpm[1];
        state_old.rpm[2] = state_new.rpm[2];
        state_old.rpm[3] = state_new.rpm[3];
        state_old.force[0] = state_new.force[0];
        state_old.force[1] = state_new.force[1];
        state_old.force[2] = state_new.force[2];
        state_old.torque[0] = state_new.torque[0];
        state_old.torque[1] = state_new.torque[1];
        state_old.torque[2] = state_new.torque[2];
        for(TI step_i=0; step_i < ENVIRONMENT_OLD::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH; step_i++){
            state_old.action_history[step_i][0] = state_new.action_history[step_i][0];
            state_old.action_history[step_i][1] = state_new.action_history[step_i][1];
            state_old.action_history[step_i][2] = state_new.action_history[step_i][2];
            state_old.action_history[step_i][3] = state_new.action_history[step_i][3];
        }

        constexpr T threshold = 1e-10;
        for(TI step_i=0; step_i < 100; step_i++){
            std::cout << "Step i: " << step_i << std::endl;
            bpt_old::observe(device_old, env_old, state_old, observation_old, rng_old);
            bpt_new::observe(device_new, env_new, state_new, observation_new, rng_new);
            bpt_old::observe_privileged(device_old, env_old, state_old, observation_privileged_old, rng_old);
            bpt_new::observe_privileged(device_new, env_new, state_new, observation_privileged_new, rng_new);
            bpt_old::step(device_old, env_old, state_old, action_old, next_state_old, rng_old);
            bpt_new::step(device_new, env_new, state_new, action_new, next_state_new, rng_new);
            {
                for(TI i=0; i < ENVIRONMENT_OLD::OBSERVATION_DIM; i++){
                    std::cout << "observation i: " << i << std::endl;
                    ASSERT_NEAR(bpt_old::get(observation_old, 0, i), bpt_new::get(observation_new, 0, i), threshold);
                }
                for(TI i=0; i < ENVIRONMENT_OLD::OBSERVATION_DIM_PRIVILEGED; i++){
                    std::cout << "observation_privileged i: " << i << std::endl;
                    ASSERT_NEAR(bpt_old::get(observation_privileged_old, 0, i), bpt_new::get(observation_privileged_new, 0, i), threshold);
                }
            }
            {
                ASSERT_NEAR(next_state_old.position[0], next_state_new.position[0], threshold);
                ASSERT_NEAR(next_state_old.position[1], next_state_new.position[1], threshold);
                ASSERT_NEAR(next_state_old.position[2], next_state_new.position[2], threshold);
                ASSERT_NEAR(next_state_old.orientation[0], next_state_new.orientation[0], threshold);
                ASSERT_NEAR(next_state_old.orientation[1], next_state_new.orientation[1], threshold);
                ASSERT_NEAR(next_state_old.orientation[2], next_state_new.orientation[2], threshold);
                ASSERT_NEAR(next_state_old.orientation[3], next_state_new.orientation[3], threshold);
                ASSERT_NEAR(next_state_old.linear_velocity[0], next_state_new.linear_velocity[0], threshold);
                ASSERT_NEAR(next_state_old.linear_velocity[1], next_state_new.linear_velocity[1], threshold);
                ASSERT_NEAR(next_state_old.linear_velocity[2], next_state_new.linear_velocity[2], threshold);
                ASSERT_NEAR(next_state_old.angular_velocity[0], next_state_new.angular_velocity[0], threshold);
                ASSERT_NEAR(next_state_old.angular_velocity[1], next_state_new.angular_velocity[1], threshold);
                ASSERT_NEAR(next_state_old.angular_velocity[2], next_state_new.angular_velocity[2], threshold);
                ASSERT_NEAR(next_state_old.rpm[0], next_state_new.rpm[0], threshold);
                ASSERT_NEAR(next_state_old.rpm[1], next_state_new.rpm[1], threshold);
                ASSERT_NEAR(next_state_old.rpm[2], next_state_new.rpm[2], threshold);
                ASSERT_NEAR(next_state_old.rpm[3], next_state_new.rpm[3], threshold);
                ASSERT_NEAR(next_state_old.force[0], next_state_new.force[0], threshold);
                ASSERT_NEAR(next_state_old.force[1], next_state_new.force[1], threshold);
                ASSERT_NEAR(next_state_old.force[2], next_state_new.force[2], threshold);
                ASSERT_NEAR(next_state_old.torque[0], next_state_new.torque[0], threshold);
                ASSERT_NEAR(next_state_old.torque[1], next_state_new.torque[1], threshold);
                ASSERT_NEAR(next_state_old.torque[2], next_state_new.torque[2], threshold);
                for(TI step_i=0; step_i < ENVIRONMENT_OLD::STATIC_PARAMETERS::ACTION_HISTORY_LENGTH; step_i++){
                    ASSERT_NEAR(next_state_old.action_history[step_i][0], next_state_new.action_history[step_i][0], threshold);
                    ASSERT_NEAR(next_state_old.action_history[step_i][1], next_state_new.action_history[step_i][1], threshold);
                    ASSERT_NEAR(next_state_old.action_history[step_i][2], next_state_new.action_history[step_i][2], threshold);
                    ASSERT_NEAR(next_state_old.action_history[step_i][3], next_state_new.action_history[step_i][3], threshold);
                }
            }
            T r_new = bpt_new::reward(device_new, env_new, state_new, action_new, next_state_new, rng_new);
            T r_old = bpt_old::reward(device_old, env_old, state_old, action_old, next_state_old, rng_old);
            ASSERT_NEAR(r_new, r_old, threshold);
            state_old = next_state_old;
            state_new = next_state_new;
        }
    }


    std::cout << "end" << std::endl;
}
