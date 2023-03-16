#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>
#include <layer_in_c/rl/environments/mujoco/ant/ui.h>

namespace lic = layer_in_c;

#include <chrono>
#include <iostream>

#include <gtest/gtest.h>

namespace TEST_DEFINITIONS{
    using DEVICE = lic::devices::DefaultCPU_TENSORBOARD;
    using T = double;
    using TI = typename DEVICE::index_t;
    using ENVIRONMENT_SPEC = lic::rl::environments::mujoco::ant::Specification<T, TI, lic::rl::environments::mujoco::ant::DefaultParameters<T, TI>>;
    using ENVIRONMENT = lic::rl::environments::mujoco::Ant<ENVIRONMENT_SPEC>;
    using UI = lic::rl::environments::mujoco::ant::UI<ENVIRONMENT>;
}


TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, UI){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    UI ui;

    lic::malloc(dev, env);
    lic::init(dev, env, ui);

    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    typename ENVIRONMENT::State state, next_state;
    lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
    lic::malloc(dev, action);
    lic::set_all(dev, action, 1);
    lic::sample_initial_state(dev, env, state, rng);
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 10000; i++) {
        for (TI action_i = 0; action_i < ENVIRONMENT::ACTION_DIM; action_i++){
            set(action, 0, action_i, lic::random::uniform_real_distribution(DEVICE::SPEC::RANDOM(), -0.5, 0.5, rng));
        }
        T dt = lic::step(dev, env, state, action, next_state);
        std::this_thread::sleep_for(std::chrono::milliseconds((int)(dt*1000)));
        lic::set_state(dev, ui, state);
        state = next_state;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end-start;
    std::cout << "Time: " << diff.count() << std::endl;

}

