#include <backprop_tools/operations/cpu_tensorboard.h>

#include <backprop_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <backprop_tools/rl/environments/mujoco/ant/ui.h>

namespace bpt = backprop_tools;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

namespace TEST_DEFINITIONS{
    using DEVICE = bpt::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = bpt::rl::environments::mujoco::ant::Specification<T, TI, bpt::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
    using ENVIRONMENT = bpt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    using UI = bpt::rl::environments::mujoco::ant::UI<ENVIRONMENT>;
}


TEST(BACKPROP_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, UI){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    UI ui;

    bpt::malloc(dev, env);
    bpt::init(dev, env, ui);

    auto rng = bpt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    typename ENVIRONMENT::State state, next_state;
    bpt::MatrixDynamic<bpt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    bpt::malloc(dev, action);
    bpt::set_all(dev, action, 1);
    bpt::sample_initial_state(dev, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10000; i++) {
        for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, bpt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), -0.5, 0.5, rng));
        }
        T dt = bpt::step(dev, env, state, action, next_state);
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(dt*1000)));
        bpt::set_state(dev, ui, state);
        state = next_state;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

}

