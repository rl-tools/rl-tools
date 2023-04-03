#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>

namespace lic = layer_in_c;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>
#include <highfive/H5File.hpp>

namespace TEST_DEFINITIONS{
    using DEVICE = lic::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, lic::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
    using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
}


TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, MAIN){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    lic::malloc(dev, env);

    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    typename ENVIRONMENT::State state, next_state;
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    lic::malloc(dev, action);
    lic::set_all(dev, action, 1);
    lic::sample_initial_state(dev, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 1; i++){
        lic::step(dev, env, state, action, next_state);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

}

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, STATE_COMPLETENESS){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    lic::malloc(dev, env);

    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    using STATE = typename ENVIRONMENT::State;
    STATE initial_state, state, next_state_1, next_state_2, next_state_temp;
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> initial_action, action;
    lic::malloc(dev, initial_action);
    lic::malloc(dev, action);
    std::vector<std::vector<T>> states_q;
    std::vector<std::vector<T>> states_q_dot;
    std::vector<std::vector<T>> next_states_q;
    std::vector<std::vector<T>> next_states_q_dot;
    std::vector<std::vector<T>> actions;
    std::vector<T> rewards;
    std::vector<bool> terminated;
    for(TI episode_i = 0; episode_i < 5; episode_i++){
        lic::sample_initial_state(dev, env, state, rng);
        for(TI step_i = 0; step_i < 1000; step_i++){
            lic::randn(dev, action, rng);
            lic::clamp(dev, action, -1, 1);
            lic::step(dev, env, state, action, next_state_temp);
            {
                auto q_temp = lic::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q>(dev, (T*)state.q);
                auto q_dot_temp = lic::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q_DOT>(dev, (T*)state.q_dot);
                states_q.push_back(lic::std_vector(dev, q_temp)[0]);
                states_q_dot.push_back(lic::std_vector(dev, q_dot_temp)[0]);
                actions.push_back(lic::std_vector(dev, action)[0]);
                rewards.push_back(lic::reward(dev, env, state, action, next_state_temp));
                terminated.push_back(lic::terminated(dev, env, state, rng));
            }
            {
                auto q_temp = lic::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q>(dev, (T*)next_state_temp.q);
                auto q_dot_temp = lic::wrap<DEVICE, T, ENVIRONMENT::SPEC::STATE_DIM_Q_DOT>(dev, (T*)next_state_temp.q_dot);
                next_states_q.push_back(lic::std_vector(dev, q_temp)[0]);
                next_states_q_dot.push_back(lic::std_vector(dev, q_dot_temp)[0]);
            }
            if(episode_i == 0 && step_i == 0){
                lic::copy(dev, dev, initial_action, action);
                initial_state = state;
                next_state_1 = next_state_temp;
            }
            state = next_state_temp;
        }
    }
    state = initial_state;
    lic::step(dev, env, state, initial_action, next_state_2);

    T acc = 0;
    for(TI state_i=0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
        acc += lic::math::abs(typename DEVICE::SPEC::MATH(), next_state_1.q[state_i] - next_state_2.q[state_i]);
    }
    for(TI state_i=0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
        acc += lic::math::abs(typename DEVICE::SPEC::MATH(), next_state_1.q_dot[state_i] - next_state_2.q_dot[state_i]);
    }
    std::cout << "next_state_1 vs. next_state_2 abs diff: " << acc << std::endl;
    ASSERT_LT(acc, 1e-12);
}

TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, CHECK_INTERFACE){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    lic::malloc(dev, env);
    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    const std::string data_path = "tests_rl_environments_mujoco_ant_data.h5";
    auto data_file = HighFive::File(data_path, HighFive::File::ReadOnly);
    std::vector<std::vector<T>> observations, next_observations, states, next_states, actions;
    std::vector<T> rewards;
    std::vector<T> terminated_flags;
    std::vector<T> truncated_flags;
    data_file.getDataSet("observations").read(observations);
    data_file.getDataSet("next_observations").read(next_observations);
    data_file.getDataSet("states").read(states);
    data_file.getDataSet("next_states").read(next_states);
    data_file.getDataSet("actions").read(actions);
    data_file.getDataSet("rewards").read(rewards);
    data_file.getDataSet("terminated_flags").read(terminated_flags);
    data_file.getDataSet("truncated_flags").read(truncated_flags);

    assert(observations.size() == next_observations.size());
    assert(observations.size() == states.size());
    assert(observations.size() == next_states.size());
    assert(observations.size() == actions.size());
    assert(observations.size() == rewards.size());
    assert(observations.size() == terminated_flags.size());
    assert(observations.size() == truncated_flags.size());

    using STATE = typename ENVIRONMENT::State;
    STATE state, initial_state, next_state, termination_check_state;
    lic::MatrixDynamic<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    lic::malloc(dev, action);
    bool load_state = true;
    TI state_age = 0;
    TI episode_i = 0;
    for(TI step_i = 0; step_i < observations.size(); step_i++){
        std::cout << "step_i: " << step_i << std::endl;
        if(load_state){
            lic::sample_initial_state(dev, env, state, rng);
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
        lic::step(dev, env, state, action, next_state);
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
            T abs_diff = lic::math::abs(typename DEVICE::SPEC::MATH(), next_state.q[state_i] - next_states[step_i][state_i]);
            if(abs_diff > 0){
                T relative_diff = lic::math::abs(typename DEVICE::SPEC::MATH(), next_state.q[state_i] - initial_state.q[state_i]);
                T ratio = relative_diff / abs_diff;
                std::cout << "ratio: " << ratio << std::endl;
                if(abs_diff > 1e-10 && ratio < 1e10){
                    ASSERT_TRUE(false);
                }
            }
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
            ASSERT_NEAR(next_state.q_dot[state_i], next_states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q], 1e-10);
        }
        T reward = lic::reward(dev, env, state, action, next_state);
        T reward_abs_diff = lic::math::abs(typename DEVICE::SPEC::MATH(), reward - rewards[step_i]);
        if(reward_abs_diff > 1e-2){
            ASSERT_NEAR(reward, rewards[step_i], 1e-5);
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q; state_i++){
            termination_check_state.q[state_i] = next_states[step_i][state_i];
        }
        for(TI state_i = 0; state_i < ENVIRONMENT::SPEC::STATE_DIM_Q_DOT; state_i++){
            termination_check_state.q_dot[state_i] = next_states[step_i][state_i + ENVIRONMENT::SPEC::STATE_DIM_Q];
        }
        bool terminated_flag = lic::terminated(dev, env, termination_check_state, rng);
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
