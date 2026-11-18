#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cuda.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>

#include <gtest/gtest.h>
#include <metra/metra.h>

#include <algorithm>
#include <cmath>

namespace rlt = rl_tools;
namespace hd = rlt::rl::environments::hyperdrone;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_GPU = rlt::devices::DEVICE_FACTORY_CUDA<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using RNG_GPU = typename DEVICE_GPU::SPEC::RANDOM::ENGINE<>;
using T = float;
using TI = typename DEVICE::index_t;

namespace test_kinematic_cuda {
    struct Limits {
        static constexpr T LINEAR_VELOCITY = 2;
        static constexpr T YAW_RATE = 1.5;
    };
    struct WORLD_SPEC: hd::Specification<T, TI, hd::presets::x500_fpv::DynamicsStaticParameters<T, TI>> {
        template <typename DYNAMICS_SPEC>
        using DYNAMICS_ENV = hd::KinematicMultirotor<DYNAMICS_SPEC, Limits>;
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 37;
    };
    using WORLD = hd::World<WORLD_SPEC>;
    constexpr TI INSTANCES = WORLD::INSTANCES;
    constexpr TI STEPS = 50;
    using PARAMETERS = rlt::Tensor<rlt::tensor::Specification<typename WORLD::Parameters, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
    using STATES = rlt::Tensor<rlt::tensor::Specification<typename WORLD::State, TI, rlt::tensor::Shape<TI, INSTANCES>>>;
    using ACTIONS = rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, INSTANCES, WORLD::ACTION_DIM>>>;

    template <typename DEVICE_TAG, typename RNG_STATES>
    __global__ void rollout_kernel(DEVICE_TAG device, typename WORLD::DYNAMICS_ENV dynamics, PARAMETERS parameters, STATES initial_states, STATES final_states, const ACTIONS actions, RNG_STATES rng) {
        const TI instance_i = threadIdx.x + blockIdx.x * blockDim.x;
        if (instance_i < INSTANCES) {
            auto& rng_state = rlt::get(rng.states, 0, instance_i);
            auto& instance_parameters = rlt::get_ref(device, parameters, instance_i);
            typename WORLD::State state, next_state;
            hd::_sample_dynamics_state<DEVICE_TAG, WORLD_SPEC>(device, dynamics, instance_parameters, state, rng_state);
            rlt::get_ref(device, initial_states, instance_i) = state;
            rlt::Matrix<rlt::matrix::Specification<T, TI, 1, WORLD::ACTION_DIM, false>> action;
            for (TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++) {
                rlt::set(action, 0, action_i, rlt::get(device, actions, instance_i, action_i));
            }
            for (TI step_i = 0; step_i < STEPS; step_i++) {
                hd::_step<DEVICE_TAG, WORLD_SPEC>(device, dynamics, instance_parameters, state, action, next_state, rng_state);
                state = next_state;
            }
            rlt::get_ref(device, final_states, instance_i) = state;
        }
    }
}

// the device samples level starts at rest and rolls them out like the host
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_KINEMATIC_CUDA, DEVICE_ROLLOUT_MATCHES_HOST) {
    using namespace test_kinematic_cuda;
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        GTEST_SKIP() << "CUDA device unavailable";
    }
    DEVICE device;
    DEVICE_GPU device_gpu;
    rlt::init(device);
    rlt::init(device_gpu);
    WORLD world;
    PARAMETERS parameters, parameters_gpu;
    STATES initial_states, initial_states_gpu, final_states, final_states_gpu;
    ACTIONS actions, actions_gpu;
    rlt::malloc(device, parameters);
    rlt::malloc(device_gpu, parameters_gpu);
    rlt::malloc(device, initial_states);
    rlt::malloc(device_gpu, initial_states_gpu);
    rlt::malloc(device, final_states);
    rlt::malloc(device_gpu, final_states_gpu);
    rlt::malloc(device, actions);
    rlt::malloc(device_gpu, actions_gpu);
    for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
        rlt::get_ref(device, parameters, instance_i) = world.parameters;
        for (TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++) {
            rlt::set(device, actions, (T)(1.2 * std::sin(1.3 * instance_i + 2.1 * action_i)), instance_i, action_i);
        }
    }
    rlt::copy(device, device_gpu, parameters, parameters_gpu);
    rlt::copy(device, device_gpu, actions, actions_gpu);
    RNG_GPU rng_gpu;
    rlt::malloc(device_gpu, rng_gpu);
    rlt::init(device_gpu, rng_gpu, 1337);
    constexpr TI BLOCK_SIZE = 32;
    rlt::devices::cuda::TAG<DEVICE_GPU, true> tag_device{};
    rollout_kernel<<<(INSTANCES + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, device_gpu.stream>>>(tag_device, world.dynamics, parameters_gpu, initial_states_gpu, final_states_gpu, actions_gpu, rng_gpu);
    rlt::check_status(device_gpu);
    rlt::copy(device_gpu, device, initial_states_gpu, initial_states);
    rlt::copy(device_gpu, device, final_states_gpu, final_states);

    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 1337);
    double max_error = 0;
    for (TI instance_i = 0; instance_i < INSTANCES; instance_i++) {
        auto& instance_parameters = rlt::get_ref(device, parameters, instance_i);
        const auto& init = instance_parameters.agents[0].dynamics.mdp.init;
        typename WORLD::State state = rlt::get(device, initial_states, instance_i);
        typename WORLD::State next_state;
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            EXPECT_LE(std::abs(state.position[axis_i] - init.position[axis_i]), init.max_position + 1e-6);
            EXPECT_EQ(state.linear_velocity[axis_i], 0);
            EXPECT_EQ(state.angular_velocity[axis_i], 0);
        }
        EXPECT_EQ(state.orientation[1], 0);
        EXPECT_EQ(state.orientation[2], 0);
        EXPECT_LE(std::abs(2 * std::atan2(state.orientation[3], state.orientation[0])), init.max_angle + 1e-5);
        rlt::Matrix<rlt::matrix::Specification<T, TI, 1, WORLD::ACTION_DIM, false>> action;
        for (TI action_i = 0; action_i < WORLD::ACTION_DIM; action_i++) {
            rlt::set(action, 0, action_i, rlt::get(device, actions, instance_i, action_i));
        }
        for (TI step_i = 0; step_i < STEPS; step_i++) {
            hd::_step<DEVICE, WORLD_SPEC>(device, world.dynamics, instance_parameters, state, action, next_state, rng);
            state = next_state;
        }
        const auto& device_state = rlt::get_ref(device, final_states, instance_i);
        for (TI axis_i = 0; axis_i < 3; axis_i++) {
            EXPECT_NEAR(device_state.position[axis_i], state.position[axis_i], 1e-5);
            EXPECT_NEAR(device_state.linear_velocity[axis_i], state.linear_velocity[axis_i], 1e-5);
            EXPECT_NEAR(device_state.angular_velocity[axis_i], state.angular_velocity[axis_i], 1e-6);
            max_error = std::max(max_error, (double)std::abs(device_state.position[axis_i] - state.position[axis_i]));
        }
        for (TI component_i = 0; component_i < 4; component_i++) {
            EXPECT_NEAR(device_state.orientation[component_i], state.orientation[component_i], 1e-5);
        }
    }
    metra::log("hyperdrone/kinematic/cuda/host_device_position_max_error", max_error);
    rlt::free(device, rng);
    rlt::free(device_gpu, rng_gpu);
    rlt::free(device, parameters);
    rlt::free(device_gpu, parameters_gpu);
    rlt::free(device, initial_states);
    rlt::free(device_gpu, initial_states_gpu);
    rlt::free(device, final_states);
    rlt::free(device_gpu, final_states_gpu);
    rlt::free(device, actions);
    rlt::free(device_gpu, actions_gpu);
    rlt::free(device_gpu);
}
