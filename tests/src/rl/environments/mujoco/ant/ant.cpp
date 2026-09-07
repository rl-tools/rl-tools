#include <rl_tools/operations/cpu.h>

#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>

#include "../../../../utils/utils.h"

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>
#include <rl_tools/persist/backends/tar/operations_cpu.h>
#include <rl_tools/rl/environments/mujoco/ant/persist.h>
#include <metra/metra.h>

namespace TEST_DEFINITIONS{
    using DEVICE = rlt::devices::DefaultCPU;
    using RNG = typename DEVICE::SPEC::RANDOM::ENGINE<>;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = rlt::rl::environments::mujoco::ant::Specification<T, TI, rlt::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
    using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
}


TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, MAIN){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters parameters;
    rlt::malloc(dev, env);

    RNG rng;
    rlt::malloc(dev, rng);
    rlt::init(dev, rng, 10);

    typename ENVIRONMENT::State state, next_state;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, action);
    rlt::set_all(dev, action, 1);
    rlt::sample_initial_parameters(dev, env, parameters, rng);
    rlt::sample_initial_state(dev, env, parameters, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        rlt::step(dev, env, parameters, state, action, next_state, rng);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, STATE_COMPLETENESS){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters env_parameters;
    rlt::malloc(dev, env);

    RNG rng;
    rlt::malloc(dev, rng);
    rlt::init(dev, rng, 10);

    using STATE = typename ENVIRONMENT::State;
    STATE initial_state, state, next_state_1, next_state_2, next_state_temp;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> initial_action, action;
    rlt::malloc(dev, initial_action);
    rlt::malloc(dev, action);
    std::vector<std::vector<T>> states_q;
    std::vector<std::vector<T>> states_q_dot;
    std::vector<std::vector<T>> next_states_q;
    std::vector<std::vector<T>> next_states_q_dot;
    std::vector<std::vector<T>> actions;
    std::vector<T> rewards;
    std::vector<bool> terminated;
    for(TI episode_i = 0; episode_i < 5; episode_i++){
        rlt::sample_initial_parameters(dev, env, env_parameters, rng);
        rlt::sample_initial_state(dev, env, env_parameters, state, rng);
        for(TI step_i = 0; step_i < 1000; step_i++){
            rlt::randn(dev, action, rng);
            rlt::clamp(dev, action, -1, 1);
            rlt::step(dev, env, env_parameters, state, action, next_state_temp, rng);
            {
                auto q_temp = rlt::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q>(dev, (T*)state.q);
                auto q_dot_temp = rlt::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q_DOT>(dev, (T*)state.q_dot);
                states_q.push_back(rlt::std_vector(dev, q_temp)[0]);
                states_q_dot.push_back(rlt::std_vector(dev, q_dot_temp)[0]);
                actions.push_back(rlt::std_vector(dev, action)[0]);
                rewards.push_back(rlt::reward(dev, env, env_parameters, state, action, next_state_temp, rng));
                terminated.push_back(rlt::terminated(dev, env, env_parameters, state, rng));
            }
            {
                auto q_temp = rlt::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q>(dev, (T*)next_state_temp.q);
                auto q_dot_temp = rlt::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q_DOT>(dev, (T*)next_state_temp.q_dot);
                next_states_q.push_back(rlt::std_vector(dev, q_temp)[0]);
                next_states_q_dot.push_back(rlt::std_vector(dev, q_dot_temp)[0]);
            }
            if(episode_i == 0 && step_i == 0){
                rlt::copy(dev, dev, action, initial_action);
                initial_state = state;
                next_state_1 = next_state_temp;
            }
            state = next_state_temp;
        }
    }
    state = initial_state;
    rlt::step(dev, env, env_parameters, state, initial_action, next_state_2, rng);

    T acc = 0;
    for(TI state_i=0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
        acc += rlt::math::abs(typename DEVICE::SPEC::MATH(), next_state_1.q[state_i] - next_state_2.q[state_i]);
    }
    for(TI state_i=0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
        acc += rlt::math::abs(typename DEVICE::SPEC::MATH(), next_state_1.q_dot[state_i] - next_state_2.q_dot[state_i]);
    }
    std::cout << "next_state_1 vs. next_state_2 abs diff: " << acc << std::endl;
    ASSERT_LT(acc, 1e-12);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, CHECK_INTERFACE){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    ENVIRONMENT::Parameters env_parameters;
    rlt::malloc(dev, env);
    RNG rng;
    rlt::malloc(dev, rng);
    rlt::init(dev, rng, 10);

    std::string DATA_FILE_NAME = "tests_rl_environments_mujoco_ant_data.h5";
    const char *data_path_stub = RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH);
    std::string DATA_FILE_PATH = std::string(data_path_stub) + "/" + DATA_FILE_NAME;
    auto data_file = rl_tools::persist::backends::hdf5::File(DATA_FILE_PATH, rl_tools::persist::backends::hdf5::Mode::READ);
    std::vector<std::vector<T>> observations, next_observations, states, next_states, actions;
    std::vector<T> rewards;
    std::vector<T> terminated_flags;
    std::vector<T> truncated_flags;
    rl_tools::persist::backends::hdf5::Group<> data_file_root{H5Gopen2(data_file.id, ".", H5P_DEFAULT)};
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "observations", observations);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "next_observations", next_observations);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "states", states);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "next_states", next_states);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "actions", actions);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "rewards", rewards);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "terminated_flags", terminated_flags);
    rl_tools::persist::backends::hdf5::read_dataset(data_file_root, "truncated_flags", truncated_flags);

    assert(observations.size() == next_observations.size());
    assert(observations.size() == states.size());
    assert(observations.size() == next_states.size());
    assert(observations.size() == actions.size());
    assert(observations.size() == rewards.size());
    assert(observations.size() == terminated_flags.size());
    assert(observations.size() == truncated_flags.size());

    using STATE = typename ENVIRONMENT::State;
    STATE state, initial_state, next_state, termination_check_state;
    rlt::Matrix<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, action);
    bool load_state = true;
    TI state_age = 0;
    TI episode_i = 0;
    for(TI step_i = 0; step_i < observations.size(); step_i++){
        std::cout << "step_i: " << step_i << std::endl;
        if(load_state){
            rlt::sample_initial_parameters(dev, env, env_parameters, rng);
            rlt::sample_initial_state(dev, env, env_parameters, state, rng);
            for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
                state.q[state_i] = states[step_i][state_i];
                env.data->qpos[state_i] = states[step_i][state_i];
            }
            for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
                state.q_dot[state_i] = states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q];
                env.data->qvel[state_i] = states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q];
            }
            mj_forward(env.model, env.data);
            load_state = false;
            state_age = 0;
            initial_state = state;
        }
        for(TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, actions[step_i][action_i]);
        }
        mj_forward(env.model, env.data);
        rlt::step(dev, env, env_parameters, state, action, next_state, rng);
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
            T abs_diff = rlt::math::abs(typename DEVICE::SPEC::MATH(), next_state.q[state_i] - next_states[step_i][state_i]);
            if(abs_diff > 0){
                T relative_diff = rlt::math::abs(typename DEVICE::SPEC::MATH(), next_state.q[state_i] - initial_state.q[state_i]);
                T ratio = relative_diff / abs_diff;
                std::cout << "ratio: " << ratio << std::endl;
                if(abs_diff > 1e-10 && ratio < 1e10){
                    ASSERT_TRUE(false);
                }
            }
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
            ASSERT_NEAR(next_state.q_dot[state_i], next_states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q], 1e-9);
        }
        T reward = rlt::reward(dev, env, env_parameters, state, action, next_state, rng);
        T reward_abs_diff = rlt::math::abs(typename DEVICE::SPEC::MATH(), reward - rewards[step_i]);
        if(reward_abs_diff > 1e-2){
            ASSERT_NEAR(reward, rewards[step_i], 1e-5);
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
            termination_check_state.q[state_i] = next_states[step_i][state_i];
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
            termination_check_state.q_dot[state_i] = next_states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q];
        }
        bool terminated_flag = rlt::terminated(dev, env, env_parameters, termination_check_state, rng);
        assert(terminated_flag == (terminated_flags[step_i] == 1));
        bool truncated_flag = (episode_i == 999);
        assert(truncated_flag == (truncated_flags[step_i] == 1));
        if(truncated_flag || terminated_flag){
            episode_i = 0;
        }
        else{
            episode_i++;
        }
        if(terminated_flag || truncated_flag || state_age > 30){
            load_state = true;
        }
        else{
            state = next_state;
            state_age++;
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, SEMANTIC_CHECKPOINT_RESUME){
    using namespace rl_tools;
    using namespace TEST_DEFINITIONS;
    DEVICE device;
    ENVIRONMENT original, hdf5, tar;
    RNG rng;
    malloc(device, original); malloc(device, hdf5); malloc(device, tar);
    malloc(device, rng); init(device, rng, 42);
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State state, next;
    Matrix<matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> actions;
    malloc(device, actions); set_all(device, actions, (T)0.3);
    original.model->opt.gravity[2] = -5;
    sample_initial_parameters(device, original, parameters, rng);
    sample_initial_state(device, original, parameters, state, rng);
    for(TI i = 0; i < 5; i++){ step(device, original, parameters, state, actions, next, rng); state = next; }
    const char* path = "test_ant_environment_checkpoint.h5";
    {
        persist::backends::hdf5::File file(path, persist::backends::hdf5::Mode::WRITE);
        auto group = create_group(device, file, "environment");
        save(device, original, group);
    }
    {
        persist::backends::hdf5::File file(path, persist::backends::hdf5::Mode::READ);
        auto group = get_group(device, file, "environment");
        ASSERT_TRUE(load(device, hdf5, group));
    }
    persist::backends::tar::Writer writer;
    persist::backends::tar::WriterGroup<persist::backends::tar::WriterGroupSpecification<TI, decltype(writer)>> output{"", &writer};
    save(device, original, output);
    persist::backends::tar::finalize(device, writer);
    persist::backends::tar::ReaderGroup<persist::backends::tar::ReaderGroupSpecification<TI>> input;
    input.data = {writer.buffer.data(), static_cast<TI>(writer.buffer.size())};
    ASSERT_TRUE(load(device, tar, input));
    EXPECT_NE(hdf5.model, original.model); EXPECT_NE(tar.model, original.model);
    EXPECT_NE(hdf5.data, original.data); EXPECT_NE(tar.data, original.data);
    EXPECT_EQ(hdf5.model->opt.gravity[2], -5); EXPECT_EQ(tar.model->opt.gravity[2], -5);
    for(TI i = 0; i < 10; i++){
        ENVIRONMENT::State hdf5_next, tar_next;
        step(device, original, parameters, state, actions, next, rng);
        step(device, hdf5, parameters, state, actions, hdf5_next, rng);
        step(device, tar, parameters, state, actions, tar_next, rng);
        for(TI j = 0; j < ENVIRONMENT_SPEC::STATE_DIM_Q; j++){
            EXPECT_EQ(next.q[j], hdf5_next.q[j]); EXPECT_EQ(next.q[j], tar_next.q[j]);
        }
        for(TI j = 0; j < ENVIRONMENT_SPEC::STATE_DIM_Q_DOT; j++){
            EXPECT_EQ(next.q_dot[j], hdf5_next.q_dot[j]); EXPECT_EQ(next.q_dot[j], tar_next.q_dot[j]);
        }
        EXPECT_EQ(original.last_reward, hdf5.last_reward); EXPECT_EQ(original.last_reward, tar.last_reward);
        EXPECT_EQ(original.last_terminated, hdf5.last_terminated); EXPECT_EQ(original.last_terminated, tar.last_terminated);
        state = next;
    }
    free(device, actions); free(device, rng); free(device, tar); free(device, hdf5); free(device, original);
    std::remove(path);
    metra::log("mujoco/ant/checkpoint_failures", ::testing::Test::HasFailure() ? 1.0 : 0.0);
}
