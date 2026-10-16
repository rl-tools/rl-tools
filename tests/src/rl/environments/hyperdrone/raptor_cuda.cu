#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/raptor/operations_cuda.h>
#include <rl_tools/rl/environments/l2f/parameters/default.h>
#include <gtest/gtest.h>
#include <metra/metra.h>

namespace rlt = rl_tools;
using namespace rl_tools;
using CPU = rlt::devices::DefaultCPU;
using GPU = rlt::devices::DefaultCUDA;
using TI = GPU::index_t;
using T = float;
constexpr TI BATCH = 38;
using CONTROLLER = rlt::rl::environments::hyperdrone::controllers::raptor::Config::Bind<rlt::numeric_types::Policy<T>, TI, BATCH>;
using FACTORY = rlt::rl::environments::l2f::parameters::DEFAULT_PARAMETERS_FACTORY<T, TI, 5>;
using PARAMETERS = rlt::rl::environments::l2f::ParametersBase<FACTORY::PARAMETERS_SPEC>;
struct PhysicalState {
    T position[3] = {};
    T orientation[4] = {1, 0, 0, 0};
    T linear_velocity[3] = {};
    T angular_velocity[3] = {};
};

TEST(RL_ENVIRONMENTS_HYPERDRONE_RAPTOR_CUDA, BatchedControlAndExpandedResetMatchCPU) {
    using rlt::nn_models::sequential::content_state;
    CPU cpu;
    GPU gpu;
    CPU::SPEC::RANDOM::ENGINE<> rng;
    CONTROLLER controller_cpu, controller_gpu;
    CONTROLLER::State state_cpu, state_gpu, actual;
    CONTROLLER::Buffer buffer_cpu, buffer_gpu;
    rlt::Tensor<rlt::tensor::Specification<PARAMETERS, TI, rlt::tensor::Shape<TI, BATCH>>> parameters_cpu, parameters_gpu;
    rlt::Tensor<rlt::tensor::Specification<PhysicalState, TI, rlt::tensor::Shape<TI, BATCH>>> physical_cpu, physical_gpu;
    rlt::Tensor<rlt::tensor::Specification<T, TI, rlt::tensor::Shape<TI, BATCH, 4>>> commands_cpu, commands_gpu;
    rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, BATCH / 2>>> mask_cpu, mask_gpu;
    init(gpu);
    malloc(cpu, rng);
    init(cpu, rng, 19);
    malloc(cpu, controller_cpu);
    malloc(gpu, controller_gpu);
    malloc(cpu, state_cpu);
    malloc(gpu, state_gpu);
    malloc(cpu, actual);
    malloc(cpu, buffer_cpu);
    malloc(gpu, buffer_gpu);
    malloc(cpu, parameters_cpu);
    malloc(gpu, parameters_gpu);
    malloc(cpu, physical_cpu);
    malloc(gpu, physical_gpu);
    malloc(cpu, commands_cpu);
    malloc(gpu, commands_gpu);
    malloc(cpu, mask_cpu);
    malloc(gpu, mask_gpu);
    init_weights(cpu, controller_cpu.model, rng);
    set_all(cpu, get_layer<1>(controller_cpu.model).initial_hidden_state.parameters, T(0.25));
    copy(cpu, gpu, controller_cpu, controller_gpu);
    reset(cpu, controller_cpu, state_cpu, rng);
    reset(gpu, controller_gpu, state_gpu, rng);
    for (TI row = 0; row < BATCH; row++) {
        get_ref(cpu, parameters_cpu, row) = {};
        auto& physical = get_ref(cpu, physical_cpu, row);
        physical = {};
        physical.position[0] = T(row) / 100;
        physical.linear_velocity[1] = T(row) / 200;
    }
    for (TI row = 0; row < BATCH / 2; row++) {
        set(cpu, mask_cpu, row % 3 == 1, row);
    }
    copy(cpu, gpu, parameters_cpu, parameters_gpu);
    copy(cpu, gpu, physical_cpu, physical_gpu);
    copy(cpu, gpu, mask_cpu, mask_gpu);
    T max_error = 0;
    for (TI step_i = 0; step_i < 23; step_i++) {
        if (step_i == 11) {
            reset(cpu, controller_cpu, state_cpu, buffer_cpu, mask_cpu, rng);
            reset(gpu, controller_gpu, state_gpu, buffer_gpu, mask_gpu, rng);
            copy(gpu, cpu, state_gpu, actual);
            const auto& hidden = content_state<1>(actual.recurrent.content_state);
            for (TI row = 0; row < BATCH; row++) {
                if (row / 2 % 3 == 1) {
                    EXPECT_EQ(get(cpu, hidden.step, row), TI(0));
                    for (TI feature = 0; feature < 16; feature++) {
                        EXPECT_EQ(get(cpu, hidden.state, row, feature), T(0.25));
                    }
                    for (TI motor = 0; motor < 4; motor++) {
                        EXPECT_EQ(get(cpu, actual.previous_actions, row, motor), T(0));
                    }
                }
            }
        }
        for (TI row = 0; row < BATCH; row++) {
            for (TI component = 0; component < 4; component++) {
                set(cpu, commands_cpu, T(row + component + step_i) / 100, row, component);
            }
        }
        copy(cpu, gpu, commands_cpu, commands_gpu);
        auto actions_cpu = control(cpu, controller_cpu, state_cpu, buffer_cpu, parameters_cpu, physical_cpu, commands_cpu, rng);
        auto actions_gpu = control(gpu, controller_gpu, state_gpu, buffer_gpu, parameters_gpu, physical_gpu, commands_gpu, rng);
        EXPECT_EQ(data(actions_cpu), data(state_cpu.previous_actions));
        EXPECT_EQ(data(actions_gpu), data(state_gpu.previous_actions));
        copy(gpu, cpu, state_gpu, actual);
        for (TI row = 0; row < BATCH; row++) {
            EXPECT_EQ(get(cpu, content_state<1>(actual.recurrent.content_state).step, row), get(cpu, content_state<1>(state_cpu.recurrent.content_state).step, row));
            for (TI motor = 0; motor < 4; motor++) {
                T error = rlt::math::abs(cpu.math, get(cpu, actual.previous_actions, row, motor) - get(cpu, state_cpu.previous_actions, row, motor));
                max_error = rlt::math::max(cpu.math, max_error, error);
                EXPECT_LT(error, T(2e-5));
            }
        }
    }
    metra::log("hyperdrone/controllers/raptor/cuda_max_action_error", max_error);
    free(gpu, mask_gpu);
    free(cpu, mask_cpu);
    free(gpu, commands_gpu);
    free(cpu, commands_cpu);
    free(gpu, physical_gpu);
    free(cpu, physical_cpu);
    free(gpu, parameters_gpu);
    free(cpu, parameters_cpu);
    free(gpu, buffer_gpu);
    free(cpu, buffer_cpu);
    free(cpu, actual);
    free(gpu, state_gpu);
    free(cpu, state_cpu);
    free(gpu, controller_gpu);
    free(cpu, controller_cpu);
    free(cpu, rng);
    free(gpu);
}
