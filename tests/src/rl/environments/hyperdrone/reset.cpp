#include <rl_tools/operations/cpu.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>

#include <gtest/gtest.h>
#include <cstring>
#include <vector>

namespace rlt = rl_tools;
namespace hd = rlt::rl::environments::hyperdrone;

namespace reset_test {
    using DEVICE = rlt::devices::DefaultCPU;
    using TI = DEVICE::index_t;
    using T = float;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    struct Specification: hd::Specification<T, TI, hd::presets::x500_fpv::DynamicsStaticParameters<T, TI>> {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI MAX_DRONES = 2;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI CAM_WIDTH = 4;
        static constexpr TI CAM_HEIGHT = 4;
    };
    using Base = hd::World<Specification>;
    struct Parameters: Base::Parameters {
        T anchor[3] = {};
    };
    struct State: Base::State {
        T task_position[3] = {};
    };
    // A task layer deriving from the final placement and pose, with a controller of its own.
    struct World: Base {
        using Parameters = reset_test::Parameters;
        using State = reset_test::State;
        std::vector<int> phases;
        TI controller_resets[Base::INSTANCES] = {};
    };
    template <typename DEVICE_T, typename P, typename M, typename RNG_T>
    void reset_parameters(DEVICE_T& device, World& world, rlt::Tensor<P>& parameters, const rlt::Tensor<M>& mask, RNG_T& rng) {
        world.phases.push_back(1);
        rlt::reset_parameters(device, static_cast<Base&>(world), parameters, mask, rng);
    }
    template <typename DEVICE_T, typename P, typename S, typename M, typename RNG_T>
    void reset_state(DEVICE_T& device, World& world, rlt::Tensor<P>& parameters, rlt::Tensor<S>& states, const rlt::Tensor<M>& mask, RNG_T& rng) {
        world.phases.push_back(2);
        rlt::reset_state(device, static_cast<Base&>(world), parameters, states, mask, rng);
        for (TI i = 0; i < Base::INSTANCES; i++) {
            if (rlt::get(device, mask, i)) {
                auto& row = rlt::get_ref(device, parameters, i);
                auto& state = rlt::get_ref(device, states, i);
                for (TI axis = 0; axis < 3; axis++) {
                    row.anchor[axis] = row.scene_translation[axis];
                    state.task_position[axis] = state.position[axis];
                }
            }
        }
    }
    template <typename DEVICE_T, typename M, typename RNG_T>
    void reset_controller(DEVICE_T& device, World& world, const rlt::Tensor<M>& mask, RNG_T&) {
        world.phases.push_back(3);
        for (TI i = 0; i < Base::INSTANCES; i++) {
            if (rlt::get(device, mask, i)) {
                world.controller_resets[i]++;
            }
        }
    }

    template <typename SPEC>
    std::vector<unsigned char> bytes(DEVICE& device, const rlt::Tensor<SPEC>& tensor, TI index) {
        const auto* begin = reinterpret_cast<const unsigned char*>(&rlt::get_ref(device, tensor, index));
        return {begin, begin + sizeof(typename SPEC::T)};
    }

    struct Fixture {
        using ENVIRONMENT = rlt::rl::environments::HyperDrone<World, 2>;
        static constexpr TI TOTAL = ENVIRONMENT::INSTANCES;
        DEVICE device;
        RNG rng;
        ENVIRONMENT env;
        rlt::Tensor<rlt::tensor::Specification<World::Parameters, TI, rlt::tensor::Shape<TI, TOTAL>, false>> parameters;
        rlt::Tensor<rlt::tensor::Specification<World::State, TI, rlt::tensor::Shape<TI, TOTAL>, false>> states;
        rlt::Tensor<rlt::tensor::Specification<bool, TI, rlt::tensor::Shape<TI, TOTAL>, false>> mask;
        Fixture() {
            rlt::init(device, rng, 19);
            rlt::set_all(device, mask, true);
            for (TI member = 0; member < 2; member++) {
                auto& world = env.worlds[member];
                world.slots.resize(1);
                auto& annotations = world.slots[0].annotations;
                annotations.num_positions = 3;
                for (TI i = 0; i < annotations.num_positions; i++) {
                    annotations.positions[i].position[0] = 10 * member + i;
                    annotations.positions[i].position[1] = -(T)2 * i;
                    annotations.positions[i].position[2] = 1 + (T)0.5 * i;
                    annotations.positions[i].yaw = (T)0.2 * i;
                }
                world.num_drones = 2;
                world.history_step = 17;
                world.render_pending = false;
                rlt::malloc(device, world.render_reset);
                rlt::set_all(device, world.render_reset, false);
                rlt::malloc(device, world.drone_parameters);
                hd::DroneParameters<Specification> nominal;
                nominal.dynamics.dynamics.mass = 1;
                nominal.cameras[0].fov = 30;
                rlt::set(device, world.drone_parameters, nominal, (TI)0);
                nominal.dynamics.dynamics.mass = 2;
                nominal.cameras[0].fov = 90;
                rlt::set(device, world.drone_parameters, nominal, (TI)1);
            }
            std::memset(rlt::data(parameters), 0, sizeof(World::Parameters) * TOTAL);
            std::memset(rlt::data(states), 0, sizeof(World::State) * TOTAL);
            for (TI i = 0; i < TOTAL; i++) {
                World::Parameters row{};
                row.scene_translation[0] = -17 - (T)i;
                rlt::set(device, parameters, row, i);
                World::State state{};
                state.position[0] = 30 + (T)i;
                rlt::set(device, states, state, i);
            }
        }
        ~Fixture() {
            for (auto& world : env.worlds) {
                rlt::free(device, world.render_reset);
                rlt::free(device, world.drone_parameters);
            }
        }
        bool placed_in_scene(TI instance) const {
            const auto& row = rlt::get_ref(device, parameters, instance);
            const auto& annotations = env.worlds[instance / World::INSTANCES].slots[0].annotations;
            for (TI i = 0; i < annotations.num_positions; i++) {
                bool match = true;
                for (TI axis = 0; axis < 3; axis++) {
                    match = match && row.scene_translation[axis] == annotations.positions[i].position[axis];
                }
                if (match) {
                    return true;
                }
            }
            return false;
        }
    };
}

TEST(HyperdroneReset, ParametersDescribeTheEpisodeAfterTheFirstCall) {
    reset_test::Fixture fixture;
    using TI = reset_test::TI;
    rlt::sample_initial_parameters(fixture.device, fixture.env, fixture.parameters, fixture.mask, fixture.rng);
    for (TI i = 0; i < fixture.TOTAL; i++) {
        const auto& row = rlt::get_ref(fixture.device, fixture.parameters, i);
        EXPECT_TRUE(fixture.placed_in_scene(i)) << "instance " << i;
        EXPECT_NEAR(row.scene_yaw_cos * row.scene_yaw_cos + row.scene_yaw_sin * row.scene_yaw_sin, 1, 1e-5);
        EXPECT_LT(row.agents[0].drone, 2);
    }
    for (auto& world : fixture.env.worlds) {
        EXPECT_EQ(world.phases, (std::vector<int>{1}));
        EXPECT_FALSE(world.render_pending);
    }
}

TEST(HyperdroneReset, EditsBetweenTheCallsReachEveryDerivedState) {
    reset_test::Fixture fixture;
    using TI = reset_test::TI;
    using T = reset_test::T;
    rlt::sample_initial_parameters(fixture.device, fixture.env, fixture.parameters, fixture.mask, fixture.rng);
    for (TI i = 0; i < fixture.TOTAL; i++) {
        auto& init = rlt::get_ref(fixture.device, fixture.parameters, i).agents[0].dynamics.mdp.init;
        init.position[0] = 1 + (T)i;
        init.position[1] = -2;
        init.position[2] = 3;
        init.orientation[0] = 0;
        init.orientation[1] = 0;
        init.orientation[2] = 0;
        init.orientation[3] = 1;
        init.max_position = 0;
        init.max_angle = 0;
    }
    rlt::sample_initial_state(fixture.device, fixture.env, fixture.parameters, fixture.states, fixture.mask, fixture.rng);
    for (TI i = 0; i < fixture.TOTAL; i++) {
        const auto& state = rlt::get_ref(fixture.device, fixture.states, i);
        EXPECT_EQ(state.position[0], 1 + (T)i);
        EXPECT_EQ(state.position[1], -2);
        EXPECT_EQ(state.position[2], 3);
        EXPECT_EQ(state.orientation[0], 0);
        EXPECT_EQ(state.orientation[3], 1);
        const auto& row = rlt::get_ref(fixture.device, fixture.parameters, i);
        for (TI axis = 0; axis < 3; axis++) {
            EXPECT_EQ(state.task_position[axis], state.position[axis]) << "the task layer derives from the final pose";
            EXPECT_EQ(row.anchor[axis], row.scene_translation[axis]) << "the task layer derives from the final placement";
        }
    }
    for (TI member = 0; member < 2; member++) {
        auto& world = fixture.env.worlds[member];
        EXPECT_EQ(world.phases, (std::vector<int>{1, 2, 3}));
        EXPECT_TRUE(world.render_pending);
        EXPECT_EQ(world.history_step, 17);
        for (TI i = 0; i < reset_test::World::INSTANCES; i++) {
            EXPECT_EQ(world.controller_resets[i], 1) << "complete_reset dispatches on the most-derived World";
            EXPECT_TRUE(rlt::get(fixture.device, world.render_reset, i));
        }
    }
}

TEST(HyperdroneReset, MaskSelectsInstancesAndWorlds) {
    reset_test::Fixture fixture;
    using TI = reset_test::TI;
    rlt::set_all(fixture.device, fixture.mask, false);
    rlt::set(fixture.device, fixture.mask, true, (TI)3);
    std::vector<std::vector<unsigned char>> parameters_before, states_before;
    for (TI i = 0; i < fixture.TOTAL; i++) {
        parameters_before.push_back(reset_test::bytes(fixture.device, fixture.parameters, i));
        states_before.push_back(reset_test::bytes(fixture.device, fixture.states, i));
    }
    rlt::sample_initial_parameters(fixture.device, fixture.env, fixture.parameters, fixture.mask, fixture.rng);
    rlt::sample_initial_state(fixture.device, fixture.env, fixture.parameters, fixture.states, fixture.mask, fixture.rng);
    for (TI i = 0; i < 3; i++) {
        EXPECT_EQ(reset_test::bytes(fixture.device, fixture.parameters, i), parameters_before[i]) << "instance " << i;
        EXPECT_EQ(reset_test::bytes(fixture.device, fixture.states, i), states_before[i]) << "instance " << i;
    }
    EXPECT_NE(reset_test::bytes(fixture.device, fixture.parameters, (TI)3), parameters_before[3]);
    EXPECT_NE(reset_test::bytes(fixture.device, fixture.states, (TI)3), states_before[3]);
    EXPECT_TRUE(fixture.placed_in_scene(3));
    EXPECT_EQ(fixture.env.worlds[0].phases, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(fixture.env.worlds[1].phases, (std::vector<int>{1, 2, 3}));
    EXPECT_EQ(fixture.env.worlds[0].controller_resets[0], 0);
    EXPECT_EQ(fixture.env.worlds[0].controller_resets[1], 0);
    EXPECT_EQ(fixture.env.worlds[1].controller_resets[0], 0);
    EXPECT_EQ(fixture.env.worlds[1].controller_resets[1], 1);
    EXPECT_FALSE(rlt::get(fixture.device, fixture.env.worlds[1].render_reset, (TI)0));
    EXPECT_TRUE(rlt::get(fixture.device, fixture.env.worlds[1].render_reset, (TI)1));
}

TEST(HyperdroneReset, SampledResetsAreDeterministic) {
    reset_test::Fixture first, second;
    using TI = reset_test::TI;
    rlt::sample_initial_parameters(first.device, first.env, first.parameters, first.mask, first.rng);
    rlt::sample_initial_state(first.device, first.env, first.parameters, first.states, first.mask, first.rng);
    rlt::sample_initial_parameters(second.device, second.env, second.parameters, second.mask, second.rng);
    rlt::sample_initial_state(second.device, second.env, second.parameters, second.states, second.mask, second.rng);
    for (TI i = 0; i < first.TOTAL; i++) {
        EXPECT_EQ(reset_test::bytes(first.device, first.parameters, i), reset_test::bytes(second.device, second.parameters, i));
        EXPECT_EQ(reset_test::bytes(first.device, first.states, i), reset_test::bytes(second.device, second.states, i));
    }
}

TEST(HyperdroneReset, SpawnPinsTheEpisodeStart) {
    reset_test::Fixture fixture;
    using TI = reset_test::TI;
    using T = reset_test::T;
    using SPAWN = hd::Spawn<reset_test::Specification>;
    rlt::Tensor<rlt::tensor::Specification<SPAWN, TI, rlt::tensor::Shape<TI, reset_test::Fixture::TOTAL>, false>> spawns;
    for (TI i = 0; i < fixture.TOTAL; i++) {
        SPAWN spawn{};
        spawn.drones[0] = 1;
        spawn.translation[0] = 100 + (T)i;
        spawn.translation[1] = 5;
        spawn.translation[2] = -1;
        spawn.yaw[0] = 0;
        spawn.yaw[1] = 1;
        spawn.poses[0][0] = 1 + (T)i;
        spawn.poses[0][1] = -2;
        spawn.poses[0][2] = 3;
        spawn.poses[0][3] = 0;
        spawn.poses[0][6] = 1;
        spawn.shutter = (T)0.4;
        rlt::set(fixture.device, spawns, spawn, i);
    }
    rlt::set_all(fixture.device, fixture.mask, false);
    rlt::set(fixture.device, fixture.mask, true, (TI)1);
    rlt::set(fixture.device, fixture.mask, true, (TI)2);
    rlt::sample_initial_parameters(fixture.device, fixture.env, fixture.parameters, fixture.mask, fixture.rng);
    rlt::spawn(fixture.device, fixture.env, fixture.parameters, fixture.mask, spawns);
    rlt::sample_initial_state(fixture.device, fixture.env, fixture.parameters, fixture.states, fixture.mask, fixture.rng);
    for (TI i = 1; i < 3; i++) {
        const auto& spawn = rlt::get_ref(fixture.device, spawns, i);
        const auto& row = rlt::get_ref(fixture.device, fixture.parameters, i);
        const auto& state = rlt::get_ref(fixture.device, fixture.states, i);
        EXPECT_EQ(row.agents[0].drone, 1);
        EXPECT_EQ(row.agents[0].dynamics.dynamics.mass, 2) << "the drone-table nominal";
        EXPECT_EQ(row.agents[0].cameras[0].fov, 90);
        EXPECT_EQ(row.brightness_scale, 1);
        EXPECT_EQ(row.shutter_fraction, spawn.shutter);
        EXPECT_EQ(row.scene_yaw_cos, spawn.yaw[0]);
        EXPECT_EQ(row.scene_yaw_sin, spawn.yaw[1]);
        for (TI axis = 0; axis < 3; axis++) {
            EXPECT_EQ(row.scene_translation[axis], spawn.translation[axis]);
            EXPECT_EQ(row.anchor[axis], spawn.translation[axis]);
            EXPECT_EQ(state.position[axis], spawn.poses[0][axis]);
            EXPECT_EQ(state.task_position[axis], spawn.poses[0][axis]);
        }
        for (TI component = 0; component < 4; component++) {
            EXPECT_EQ(state.orientation[component], spawn.poses[0][3 + component]);
        }
    }
    for (TI i : {(TI)0, (TI)3}) {
        EXPECT_EQ(rlt::get_ref(fixture.device, fixture.parameters, i).scene_translation[0], -17 - (T)i) << "unselected instance " << i;
        EXPECT_EQ(rlt::get_ref(fixture.device, fixture.states, i).position[0], 30 + (T)i) << "unselected instance " << i;
    }
}
