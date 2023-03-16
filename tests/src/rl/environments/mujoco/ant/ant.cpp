#include <layer_in_c/operations/cpu_tensorboard.h>

#include <layer_in_c/rl/environments/mujoco/ant/operations_cpu.h>

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
}


TEST(LAYER_IN_C_RL_ENVIRONMENTS_MUJOCO_ANT, MAIN){
    using namespace TEST_DEFINITIONS;
    DEVICE dev;
    ENVIRONMENT env;
    lic::malloc(dev, env);

    auto rng = lic::random::default_engine(DEVICE::SPEC::RANDOM(), 10);

    typename ENVIRONMENT::State state, next_state;
    lic::Matrix<lic::matrix::Specification<T, TI, 1, ENVIRONMENT::ACTION_DIM>> action;
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

