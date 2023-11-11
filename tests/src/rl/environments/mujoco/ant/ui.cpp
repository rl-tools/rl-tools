#include <rl_tools/operations/cpu_tensorboard.h>

#include <rl_tools/rl/environments/mujoco/ant/operations_cpu.h>
#include <rl_tools/rl/environments/mujoco/ant/ui.h>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

namespace TEST_DEFINITIONS{
    using DEVICE = rlt::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = rlt::rl::environments::mujoco::ant::Specification<T, TI, rlt::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
    using ENVIRONMENT = rlt::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    using UI = rlt::rl::environments::mujoco::ant::UI<ENVIRONMENT>;
}


TEST(RL_TOOLS_RL_ENVIRONMENTS_MUJOCO_ANT, UI){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    UI ui;

    rlt::malloc(dev, env);
    rlt::init(dev, env, ui);

    auto rng = rlt::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    typename ENVIRONMENT::State state, next_state;
    rlt::MatrixDynamic<rlt::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    rlt::malloc(dev, action);
    rlt::set_all(dev, action, 1);
    rlt::sample_initial_state(dev, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++) {
        for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, rlt::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), -0.5, 0.5, rng));
        }
        T dt = rlt::step(dev, env, state, action, next_state, rng);
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(dt*1000)));
        rlt::set_state(dev, ui, state);
        state = next_state;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

}

