#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/environments/operations_cpu.h>

#include "../../../utils/utils.h"
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
namespace bpt = backprop_tools;
#define DTYPE double
const DTYPE STATE_TOLERANCE = 0.00001;

TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_PENDULUM_TEST, COMPARISON) {
    using DEVICE = bpt::devices::DefaultCPU;
    typedef bpt::rl::environments::pendulum::Specification<DTYPE, DEVICE::index_t, bpt::rl::environments::pendulum::DefaultParameters<DTYPE>> PENDULUM_SPEC;
    typedef bpt::rl::environments::Pendulum<PENDULUM_SPEC> ENVIRONMENT;
    std::string DATA_FILE_NAME = "pendulum.hdf5";
    const char *data_path_stub = BACKPROP_TOOLS_MACRO_TO_STR(BACKPROP_TOOLS_TESTS_DATA_PATH);
    std::string DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;

    typename DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
    HighFive::File file(DATA_FILE_PATH, HighFive::File::ReadOnly);
    auto episodes_group = file.getGroup("episodes");
    for(int episode_i = 0; episode_i < episodes_group.getNumberObjects(); episode_i++){
        auto episode_group = episodes_group.getGroup(std::to_string(episode_i));
        std::vector<std::vector<DTYPE>> states;
        std::vector<std::vector<DTYPE>> actions;
        std::vector<DTYPE> rewards;
        std::vector<std::vector<DTYPE>> next_states;
        std::vector<std::vector<DTYPE>> observations;
        std::vector<std::vector<DTYPE>> next_observations;

        episode_group.getDataSet("states").read(states);
        episode_group.getDataSet("actions").read(actions);
        episode_group.getDataSet("rewards").read(rewards);
        episode_group.getDataSet("next_states").read(next_states);
        episode_group.getDataSet("observations").read(observations);
        episode_group.getDataSet("next_observations").read(next_observations);
        std::cout << "episode i: " << episode_i << std::endl;
        ENVIRONMENT env;
        ENVIRONMENT::State state;
        state.theta = states[0][0];
        state.theta_dot = states[0][1];
        for(int step_i = 0; step_i < states.size(); step_i++){
            std::cout << "step i: " << step_i << std::endl;
            ENVIRONMENT::State next_state;
            bpt::MatrixDynamic<bpt::matrix::Specification<DTYPE, DEVICE::index_t, 1, ENVIRONMENT::ACTION_DIM>> action;
            bpt::malloc(device, action);
            bpt::assign(device, action, actions[step_i].data());
            bpt::step(device, env, state, action, next_state);
            DTYPE r = bpt::reward(device, env, state, action, next_state);
            EXPECT_NEAR(     states[step_i][0], state.theta, STATE_TOLERANCE);
            EXPECT_NEAR(     states[step_i][1], state.theta_dot, STATE_TOLERANCE);
            EXPECT_NEAR(    rewards[step_i]   , r, STATE_TOLERANCE);
            EXPECT_NEAR(next_states[step_i][0], next_state.theta, STATE_TOLERANCE);
            EXPECT_NEAR(next_states[step_i][1], next_state.theta_dot, STATE_TOLERANCE);
            state = next_state;
        }
    }

}
