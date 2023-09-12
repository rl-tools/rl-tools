#include <backprop_tools/operations/cpu.h>

#include <backprop_tools/nn_models/operations_cpu.h>

#include <backprop_tools/rl/environments/environments.h>
#include <backprop_tools/rl/environments/multirotor/ui.h>
#include <backprop_tools/rl/environments/multirotor/multirotor.h>
#include <backprop_tools/rl/environments/multirotor/parameters/default.h>

#include <backprop_tools/rl/environments/multirotor/operations_cpu.h>

#include <backprop_tools/rl/utils/evaluation.h>
#include <backprop_tools/nn_models/persist.h>

namespace bpt = BACKPROP_TOOLS_NAMESPACE_WRAPPER ::backprop_tools;

using DTYPE = float;
#include "../multirotor_training/parameters.h"

#include <gtest/gtest.h>
#include <chrono>
#include <thread>
#include <highfive/H5File.hpp>

using DEVICE = bpt::devices::DefaultCPU;

namespace parameter_set = parameters_0;

using parameters_environment = parameter_set::environment<DEVICE, DTYPE>;
using ENVIRONMENT = typename parameters_environment::ENVIRONMENT;

using parameters_rl = parameter_set::rl<DEVICE, DTYPE, ENVIRONMENT>;


//TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_UI, TEST_UI) {
//    DEVICE::SPEC::LOGGING logger;
//    DEVICE device(logger);
//    bpt::rl::environments::multirotor::UI<ENVIRONMENT> ui;
//    ui.host = "localhost";
//    ui.port = "8080";
////    parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
//    auto parameters = parameters_environment::parameters;
//    ENVIRONMENT env({parameters});
//    ENVIRONMENT::State state, next_state;
//    std::mt19937 rng(0);
//    bpt::init(device, env, ui);
//    bpt::sample_initial_state(device, env, state, rng);
//    for(int i = 0; i < 100; i++){
//        DTYPE action[4];
//        action[0] = 1.0;
//        action[1] = 0.0;
//        action[2] = 0.0;
//        action[3] = 0.0;
//        bpt::step(device, env, state, action, next_state);
//        state = next_state;
//        bpt::set_state(device, ui, state);
////        std::this_thread::sleep_for(std::chrono::milliseconds((int)(1000 * parameters.integration.dt)));
//        std::this_thread::sleep_for(std::chrono::milliseconds((int)(100)));
//    }
//}

std::string get_actor_file_path(){
    std::string DATA_FILE_PATH = "./actor.h5";
    const char* data_file_path = std::getenv("BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_UI_ACTOR_FILE_PATH");
    if (data_file_path != NULL){
        DATA_FILE_PATH = std::string(data_file_path);
//            std::runtime_error("Environment variable BACKPROP_TOOLS_TEST_DATA_DIR not set. Skipping test.");
    }
    return DATA_FILE_PATH;
}
//TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_UI, LOAD_ACTOR) {
//    DEVICE::SPEC::LOGGING logger;
//    DEVICE device(logger);
//    bpt::rl::environments::multirotor::UI<ENVIRONMENT> ui;
//    ui.host = "localhost";
//    ui.port = "8080";
////    parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
//    auto parameters = parameters_environment::parameters;
//    ENVIRONMENT env({parameters});
//    ENVIRONMENT::State state, next_state;
//    std::mt19937 rng(0);
//    bpt::init(device, env, ui);
//
//    parameters_rl::ACTOR_NETWORK_TYPE actor;
//    bpt::malloc(device, actor);
//
//    std::string actor_output_path = get_actor_file_path();
////    if(!std::filesystem::exists(actor_output_path)){
////        std::cout << "actor.h5 not found" << std::endl;
////        return;
////    }
//    {
//        auto actor_file = HighFive::File(actor_output_path, HighFive::File::ReadOnly);
//        bpt::load(device, actor, actor_file.getGroup("actor"));
//    }
//
//
//
//    for(DEVICE::index_t episode_i = 0; episode_i < 1; episode_i++){
//        bpt::sample_initial_state(device, env, state, rng);
//        bpt::rl::utils::evaluation::State<DTYPE, typename ENVIRONMENT::State> eval_state;
//        eval_state.state = state;
//        for (DEVICE::index_t i = 0; i < 100; i++) {
//            if(bpt::evaluate_step(device, env, ui, actor, eval_state)){
//                break;
//            }
//            std::this_thread::sleep_for(std::chrono::milliseconds((int)(parameters.integration.dt * 5000)));
//        }
////        DTYPE r = bpt::evaluate<DEVICE, ENVIRONMENT, decltype(ui), decltype(actor), decltype(rng), parameters_rl::ENVIRONMENT_STEP_LIMIT, true>(device, env, ui, actor, 1, rng);
//        std::cout << "return: " << eval_state.episode_return << std::endl;
//    }
//    bpt::free(device, actor);
//}
std::string get_replay_buffer_file_path(){
    std::string DATA_FILE_PATH = "./replay_buffer.h5";
    const char* data_file_path = std::getenv("BACKPROP_TOOLS_TEST_RL_ENVIRONMENTS_MULTIROTOR_UI_REPLAY_BUFFER_FILE_PATH");
    if (data_file_path != NULL){
        DATA_FILE_PATH = std::string(data_file_path);
//            std::runtime_error("Environment variable BACKPROP_TOOLS_TEST_DATA_DIR not set. Skipping test.");
    }
    return DATA_FILE_PATH;
}
TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MULTIROTOR_UI, LOAD_REPLAY_BUFFER) {
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = &logger;
//    parameters.mdp.init = bpt::rl::environments::multirotor::parameters::init::all_around<DTYPE, DEVICE::index_t, 4, REWARD_FUNCTION>;
    auto parameters = parameters_environment::parameters;
    std::mt19937 rng(0);

    std::string rb_output_path = get_replay_buffer_file_path();
//    if(!std::filesystem::exists(actor_output_path)){
//        std::cout << "actor.h5 not found" << std::endl;
//        return;
//    }
    std::vector<std::vector<DTYPE>> observations;
    std::vector<std::vector<DTYPE>> actions;
    std::vector<std::vector<DTYPE>> next_observations;
    std::vector<std::vector<DTYPE>> rewards_matrix;
    std::vector<std::vector<DTYPE>> terminated_matrix;
    std::vector<std::vector<DTYPE>> truncated_matrix;

    {
        auto rb_file = HighFive::File(rb_output_path, HighFive::File::ReadOnly);
        auto group = rb_file.getGroup("replay_buffer");
        group.getDataSet("observations").read(observations);
        group.getDataSet("actions").read(actions);
        group.getDataSet("next_observations").read(next_observations);
        group.getDataSet("rewards").read(rewards_matrix);
        group.getDataSet("terminated").read(terminated_matrix);
        group.getDataSet("truncated").read(truncated_matrix);
    }
    auto rewards = rewards_matrix[0];
    auto terminated = terminated_matrix[0];
    auto truncated = truncated_matrix[0];


    std::vector<std::vector<std::vector<DTYPE>>> observations_trajectories;
    std::vector<std::vector<std::vector<DTYPE>>> actions_trajectories;
    std::vector<std::vector<std::vector<DTYPE>>> next_observations_trajectories;
    std::vector<std::vector<DTYPE>> rewards_trajectories;
    std::vector<std::vector<DTYPE>> terminated_trajectories;
    std::vector<std::vector<DTYPE>> truncated_trajectories;

    for(DEVICE::index_t step_i = 0; step_i < observations.size(); step_i++){
        if(step_i == 0 || terminated[step_i - 1] == 1){
            observations_trajectories.push_back({});
            actions_trajectories.push_back({});
            next_observations_trajectories.push_back({});
            rewards_trajectories.push_back({});
            terminated_trajectories.push_back({});
            truncated_trajectories.push_back({});
        }
        observations_trajectories.back().push_back(observations[step_i]);
        actions_trajectories.back().push_back(actions[step_i]);
        next_observations_trajectories.back().push_back(next_observations[step_i]);
        rewards_trajectories.back().push_back({rewards[step_i]});
        terminated_trajectories.back().push_back({terminated[step_i]});
        truncated_trajectories.back().push_back({truncated[step_i]});
    }

    constexpr DEVICE::index_t TRAJECTORY_COUNT = 100;
    constexpr DEVICE::index_t GRID_WIDTH = 10;
    constexpr DTYPE GRID_SPACING = 1;

    ENVIRONMENT envs[TRAJECTORY_COUNT] = {parameters};
    bpt::rl::environments::multirotor::UI<ENVIRONMENT> uis[TRAJECTORY_COUNT];
    DEVICE::index_t current_trajectory[TRAJECTORY_COUNT];
    for(DEVICE::index_t trajectory_i = 0; trajectory_i < TRAJECTORY_COUNT; trajectory_i++){
        current_trajectory[trajectory_i] = trajectory_i;
        uis[trajectory_i].host = "localhost";
        uis[trajectory_i].port = "8080";
        uis[trajectory_i].id = std::to_string(trajectory_i);
        uis[trajectory_i].origin[0] = ((trajectory_i / GRID_WIDTH - ((GRID_WIDTH-1) / 2.0)) * GRID_SPACING);
        uis[trajectory_i].origin[1] = (trajectory_i % GRID_WIDTH - ((GRID_WIDTH-1) / 2.0)) * GRID_SPACING;
        uis[trajectory_i].origin[2] = 0;
        bpt::init(device, envs[trajectory_i], uis[trajectory_i]);
    }

    DEVICE::index_t current_global_trajectory = TRAJECTORY_COUNT;

    while(current_global_trajectory < observations_trajectories.size()){
        for(DEVICE::index_t trajectory_i = 0; trajectory_i < TRAJECTORY_COUNT; trajectory_i++){
            while(observations_trajectories[current_trajectory[trajectory_i]].size() == 0){
                current_trajectory[trajectory_i] = current_global_trajectory;
                current_global_trajectory++;
                if(current_global_trajectory >= observations_trajectories.size()){
                    break;
                }
            }
            if(observations_trajectories[current_trajectory[trajectory_i]].size() == 0){
                break;
            }
            auto& env = envs[trajectory_i];
            auto& ui = uis[trajectory_i];
            ENVIRONMENT::State state;
            std::vector<DTYPE> observation = observations_trajectories[current_trajectory[trajectory_i]][0];
            observations_trajectories[current_trajectory[trajectory_i]].erase(observations_trajectories[current_trajectory[trajectory_i]].begin());
            for(DEVICE::index_t i = 0; i < ENVIRONMENT::State::DIM; i++){
                state.state[i] = observation[i];
            }
            bpt::set_state(device, ui, state);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(10)));
    }
}
