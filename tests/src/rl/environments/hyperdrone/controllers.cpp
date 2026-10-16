#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/raptor/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/moving_gate/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <gtest/gtest.h>

namespace hd = rl_tools::rl::environments::hyperdrone;
using namespace rl_tools;
using DEVICE = devices::DEVICE_FACTORY<>;
using T = float;
using TI = DEVICE::index_t;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;

namespace {
    struct PositionSpec: hd::presets::X500FPVIMU<T, TI> {
        using CONTROLLER = hd::controllers::raptor::Config;
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI CAM_WIDTH = 8;
        static constexpr TI CAM_HEIGHT = 8;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 1;
    };
    using PositionWorld = hd::World<PositionSpec>;
    using TargetWorld = hd::tasks::target_frame::World<hd::tasks::target_frame::Specification<PositionWorld>>;
    using GateWorld = hd::tasks::moving_gate::World<hd::tasks::moving_gate::Specification<PositionWorld>>;
    struct DirectSpec: PositionSpec {
        using CONTROLLER = hd::controllers::direct::Config<>;
    };
    using DirectWorld = hd::World<DirectSpec>;
    using VILWorld = hd::tasks::visual_inertial_localization::World<hd::tasks::visual_inertial_localization::Specification<DirectWorld>>;

    template <typename WORLD>
    void allocate_controller(DEVICE& device, WORLD& world, RNG& rng) {
        malloc(device, world.controller);
        malloc(device, world.controller_state);
        malloc(device, world.controller_buffer);
        init_weights(device, world.controller.model, rng);
        set_all(device, nn_models::sequential::layer<1>(world.controller.model).initial_hidden_state.parameters, T(0.25));
        reset(device, world.controller, world.controller_state, rng);
        set_all(device, world.controller_buffer.observations, T(0.1));
        evaluate_step(device, world.controller, world.controller_state, world.controller_buffer, rng);
    }
    template <typename WORLD>
    void free_controller(DEVICE& device, WORLD& world) {
        free(device, world.controller_buffer);
        free(device, world.controller_state);
        free(device, world.controller);
    }
    template <typename WORLD>
    void check_task_reset() {
        DEVICE device;
        RNG rng;
        malloc(device, rng);
        init(device, rng, 42);
        WORLD world;
        allocate_controller(device, world, rng);
        init(device, world.dynamics);
        world.slots.emplace_back();
        world.slots[0].annotations.num_positions = 1;
        world.slots[0].annotations.positions[0] = {{3, -2, 1}, 0, 1};
        malloc(device, world.render_reset);
        set_all(device, world.render_reset, false);
        Tensor<tensor::Specification<typename WORLD::Parameters, TI, tensor::Shape<TI, 2>>> parameters;
        Tensor<tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, 2>>> states;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 2>>> mask;
        malloc(device, parameters);
        malloc(device, states);
        malloc(device, mask);
        for (TI instance = 0; instance < 2; instance++) {
            get_ref(device, parameters, instance) = {};
            get_ref(device, states, instance) = {};
            get_ref(device, parameters, instance).agents[0].dynamics = world.dynamics.parameters;
            get_ref(device, states, instance).position[0] = T(123);
        }
        set(device, mask, true, 0);
        set(device, mask, false, 1);
        T retained[4];
        for (TI motor = 0; motor < 4; motor++) {
            retained[motor] = get(device, world.controller_state.previous_actions, 1, motor);
        }
        sample_initial_state(device, world, parameters, states, mask, rng);
        auto& hidden = nn_models::sequential::content_state<1>(world.controller_state.recurrent.content_state);
        EXPECT_EQ(get(device, hidden.step, 0), 0);
        EXPECT_EQ(get(device, hidden.step, 1), 1);
        for (TI feature = 0; feature < 16; feature++) {
            EXPECT_EQ(get(device, hidden.state, 0, feature), T(0.25));
        }
        for (TI motor = 0; motor < 4; motor++) {
            EXPECT_EQ(get(device, world.controller_state.previous_actions, 0, motor), T(0));
            EXPECT_EQ(get(device, world.controller_state.previous_actions, 1, motor), retained[motor]);
        }
        EXPECT_EQ(get_ref(device, states, 1).position[0], T(123));
        EXPECT_TRUE(world.render_pending);
        EXPECT_TRUE(get(device, world.render_reset, 0));
        EXPECT_FALSE(get(device, world.render_reset, 1));
        free(device, mask);
        free(device, states);
        free(device, parameters);
        free(device, world.render_reset);
        free_controller(device, world);
        free(device, rng);
    }
}

TEST(HYPERDRONE_CONTROLLERS, TargetFrameResetIncludesController) {
    check_task_reset<TargetWorld>();
}
TEST(HYPERDRONE_CONTROLLERS, MovingGateResetIncludesController) {
    check_task_reset<GateWorld>();
}
TEST(HYPERDRONE_CONTROLLERS, AutonomousControllerMutationAcrossMembers) {
    DEVICE device;
    RNG rng;
    malloc(device, rng);
    init(device, rng, 42);
    hd::MultiEnvironment<VILWorld, 2> env;
    for (auto& world : env.environments) {
        allocate_controller(device, world, rng);
        world.task_step = 19;
    }
    Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 4>>> mask;
    malloc(device, mask);
    set_all(device, mask, false);
    set(device, mask, true, 2);
    reset_controller(device, env, mask, rng);
    for (TI member = 0; member < 2; member++) {
        auto& world = env.environments[member];
        auto& hidden = nn_models::sequential::content_state<1>(world.controller_state.recurrent.content_state);
        EXPECT_EQ(get(device, hidden.step, 0), member == 1 ? 0 : 1);
        EXPECT_EQ(get(device, hidden.step, 1), 1);
        EXPECT_EQ(world.task_step, 19);
        free_controller(device, world);
    }
    free(device, mask);
    free(device, rng);
}

TEST(HYPERDRONE_CONTROLLERS, AutonomousRenderInvalidationPreservesCameraPhase) {
    DEVICE device;
    VILWorld world;
    world.task_step = 19;
    ASSERT_NE(world.task_step % VILWorld::FRAME_STRIDE, 0);
    malloc(device, world.render_reset);
    Tensor<tensor::Specification<VILWorld::Parameters, TI, tensor::Shape<TI, 2>, false>> parameters;
    Tensor<tensor::Specification<VILWorld::State, TI, tensor::Shape<TI, 2>, false>> states;
    Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 2>, false>> mask;
    for (bool all : {false, true}) {
        set_all(device, world.render_reset, false);
        set(device, mask, true, 0);
        set(device, mask, all, 1);
        request_render(device, world, mask);
        render(device, world, parameters, states, mask);
        EXPECT_EQ(world.task_step, 19);
        EXPECT_FALSE(world.render_pending);
        EXPECT_TRUE(get(device, world.render_reset, 0));
        EXPECT_EQ(get(device, world.render_reset, 1), all);
    }
    free(device, world.render_reset);
}

TEST(HYPERDRONE_CONTROLLERS, SceneCommandsAndDeferredTransitionRewards) {
    DEVICE device;
    RNG rng;
    malloc(device, rng);
    init(device, rng, 3);
    PositionWorld world;
    allocate_controller(device, world, rng);
    init(device, world.dynamics);
    malloc(device, world.render_reset);
    set_all(device, world.render_reset, false);
    Tensor<tensor::Specification<PositionWorld::Parameters, TI, tensor::Shape<TI, 2>>> parameters;
    Tensor<tensor::Specification<PositionWorld::State, TI, tensor::Shape<TI, 2>>> states, next_states;
    Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 2, 4>>> commands;
    Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 2>>> rewards;
    malloc(device, parameters);
    malloc(device, states);
    malloc(device, next_states);
    malloc(device, commands);
    malloc(device, rewards);
    for (TI instance = 0; instance < 2; instance++) {
        auto& params = get_ref(device, parameters, instance);
        params = {};
        params.agents[0].dynamics = world.dynamics.parameters;
        params.scene_translation[0] = 7;
        params.scene_translation[1] = -4;
        params.scene_translation[2] = 2;
        params.scene_yaw_cos = 0;
        params.scene_yaw_sin = 1;
        initial_state(device, world.dynamics, params.agents[0].dynamics, get_ref(device, states, instance));
        set(device, commands, T(7), instance, 0);
        set(device, commands, T(-3.5), instance, 1);
        set(device, commands, T(2), instance, 2);
        set(device, commands, math::PI<T> / 2, instance, 3);
    }
    step(device, world, parameters, states, commands, next_states, rng);
    for (TI instance = 0; instance < 2; instance++) {
        EXPECT_NEAR(get(device, world.controller_buffer.observations, instance, 0), T(-0.5), 1e-6);
        EXPECT_NEAR(get(device, world.controller_buffer.observations, instance, 1), T(0), 1e-6);
        EXPECT_NEAR(get(device, world.controller_buffer.observations, instance, 3), T(1), 1e-6);
    }
    reward(device, world, parameters, states, commands, next_states, rewards, rng);
    const T before[2] = {get(device, rewards, 0), get(device, rewards, 1)};
    set_all(device, commands, T(100));
    reset(device, world.controller, world.controller_state, rng);
    reward(device, world, parameters, states, commands, next_states, rewards, rng);
    for (TI instance = 0; instance < 2; instance++) {
        EXPECT_EQ(get(device, rewards, instance), before[instance]);
    }
    free(device, rewards);
    free(device, commands);
    free(device, next_states);
    free(device, states);
    free(device, parameters);
    free(device, world.render_reset);
    free_controller(device, world);
    free(device, rng);
}

namespace contract_test {
    template <typename SPEC>
    struct Buffer: hd::controllers::direct::Buffer<SPEC> {
        Tensor<tensor::Specification<typename SPEC::T, typename SPEC::TI, tensor::Shape<typename SPEC::TI, SPEC::INSTANCES, 4>, SPEC::DYNAMIC_ALLOCATION>> dynamics_actions;
    };
    template <typename SPEC>
    struct Controller: hd::controllers::Direct<SPEC, hd::controllers::direct::Config<>::ACTION_INTERFACE> {
        static constexpr typename SPEC::TI COMMAND_DIM = 2;
        using Buffer = contract_test::Buffer<SPEC>;
    };
    struct Config: hd::controllers::direct::Config<> {
        template <typename POLICY, typename INDEX, INDEX N, bool DA = true>
        using Bind = Controller<hd::controllers::Specification<POLICY, INDEX, N, DA>>;
    };
    struct Spec: DirectSpec {
        using CONTROLLER = Config;
    };
}
namespace rl_tools {
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE& device, contract_test::Buffer<SPEC>& buffer) {
        malloc(device, buffer.dynamics_actions);
    }
    template <typename DEVICE, typename SPEC>
    void free(DEVICE& device, contract_test::Buffer<SPEC>& buffer) {
        free(device, buffer.dynamics_actions);
    }
    template <typename DEVICE, typename SPEC, typename PARAMETERS, typename STATES, typename COMMANDS, typename RNG>
    auto control(DEVICE& device, const contract_test::Controller<SPEC>&, typename contract_test::Controller<SPEC>::State&, contract_test::Buffer<SPEC>& buffer, const Tensor<PARAMETERS>&, const Tensor<STATES>&, const Tensor<COMMANDS>& commands, RNG&) {
        for (typename SPEC::TI row = 0; row < SPEC::INSTANCES; row++) {
            for (typename SPEC::TI motor = 0; motor < 4; motor++) {
                set(device, buffer.dynamics_actions, get(device, commands, row, 0) + (motor < 2 ? 1 : -1) * get(device, commands, row, 1), row, motor);
            }
        }
        return view_range(device, buffer.dynamics_actions, tensor::ViewSpec<0, SPEC::INSTANCES>{});
    }
    template <typename DEVICE, typename ACTION, typename STATE, typename INDEX>
    auto transition_action(DEVICE&, contract_test::Config, const Matrix<ACTION>&, const STATE& next, INDEX, INDEX component) {
        return next.last_action[component];
    }
}
TEST(HYPERDRONE_CONTROLLERS, CommandDimensionIsIndependentOfActuation) {
    using WORLD = hd::World<contract_test::Spec>;
    static_assert(WORLD::ACTION_DIM == 2);
    static_assert(WORLD::DYNAMICS_ACTION_DIM == 4);
    DEVICE device;
    RNG rng;
    malloc(device, rng);
    init(device, rng, 5);
    WORLD world;
    malloc(device, world.controller_buffer);
    malloc(device, world.render_reset);
    set_all(device, world.render_reset, false);
    init(device, world.dynamics);
    Tensor<tensor::Specification<WORLD::Parameters, TI, tensor::Shape<TI, 2>>> parameters;
    Tensor<tensor::Specification<WORLD::State, TI, tensor::Shape<TI, 2>>> states, next_states;
    Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 2, 2>>> commands;
    malloc(device, parameters);
    malloc(device, states);
    malloc(device, next_states);
    malloc(device, commands);
    for (TI row = 0; row < 2; row++) {
        auto& params = get_ref(device, parameters, row);
        params = {};
        params.agents[0].dynamics = world.dynamics.parameters;
        initial_state(device, world.dynamics, params.agents[0].dynamics, get_ref(device, states, row));
        set(device, commands, T(0), row, 0);
        set(device, commands, T(0.1), row, 1);
    }
    step(device, world, parameters, states, commands, next_states, rng);
    for (TI row = 0; row < 2; row++) {
        Matrix<matrix::Specification<T, TI, 1, 4, false>> dynamics_action;
        for (TI motor = 0; motor < 4; motor++) {
            set(dynamics_action, 0, motor, motor < 2 ? T(0.1) : T(-0.1));
        }
        WORLD::State expected;
        step(device, world.dynamics, get_ref(device, parameters, row).agents[0].dynamics, get_ref(device, states, row), dynamics_action, expected, rng);
        const auto& actual = get_ref(device, next_states, row);
        for (TI axis = 0; axis < 3; axis++) {
            EXPECT_EQ(actual.position[axis], expected.position[axis]);
            EXPECT_EQ(actual.angular_velocity[axis], expected.angular_velocity[axis]);
        }
    }
    free(device, commands);
    free(device, next_states);
    free(device, states);
    free(device, parameters);
    free(device, world.render_reset);
    free(device, world.controller_buffer);
    free(device, rng);
}
