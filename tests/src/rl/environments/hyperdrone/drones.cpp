#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>

#include "../../../utils/utils.h"

#include "drones_test.h"

#include <filesystem>
#include <utility>

namespace test_drones {
    struct BLUR_SPEC: WORLD_SPEC {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
        static constexpr TI N_AGENTS = 2;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = 13;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
    };
    using BLUR_WORLD = rlt::rl::environments::hyperdrone::World<BLUR_SPEC>;

    rlt::rendering::raytracing::Object quad(const char* name, float x, float y){
        rlt::rendering::raytracing::Mesh mesh;
        mesh.vertices = {x - 0.02f, y - 0.01f, 0, x + 0.02f, y - 0.01f, 0, x + 0.02f, y + 0.01f, 0, x - 0.02f, y + 0.01f, 0};
        mesh.indices = {0, 1, 2, 0, 2, 3};
        rlt::rendering::raytracing::Object object;
        object.name = name;
        object.meshes.push_back(mesh);
        return object;
    }

    struct Harness {
        DEVICE device;
        WORLD world;
        typename WORLD::SharedContext shared;
        RNG rng;
        PARAMETERS_TENSOR parameters;
        STATES_TENSOR states;
        MASK_TENSOR reset_mask;
        void init(const std::vector<DRONE>& drones, TI scenes = 1){
            rlt::init(device);
            rlt::malloc(device, shared.library);
            rlt::malloc(device, world);
            shared.drones = drones;
            rlt::rendering::datasets::procthor::GLB dataset{{}, std::vector<std::string>(scenes, SCENE_PATH)};
            typename decltype(dataset)::Corpus corpus;
            rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
            rlt::init(device, world, shared, dataset, corpus, 0, scenes, 0);
            rlt::malloc(device, rng);
            rlt::init(device, rng, 1337);
            rlt::malloc(device, parameters);
            rlt::malloc(device, states);
            rlt::malloc(device, reset_mask);
        }
        void reset(){
            rlt::set_all(device, reset_mask, true);
            rlt::sample_initial_parameters(device, world, parameters, reset_mask, rng);
            rlt::sample_initial_state(device, world, parameters, states, reset_mask, rng);
        }
        void arrange(const std::vector<TI>& drones){
            test_drones::arrange(device, parameters, states, shared.drones, drones);
        }
        void render(bool reset){
            rlt::set_all(device, reset_mask, reset);
            rlt::render(device, world, parameters, states, reset_mask);
        }
        std::vector<std::vector<uint32_t>> own_masks(){
            return test_drones::own_masks(device, device, world, shared);
        }
        void check_placements(const std::vector<TI>& drones){
            test_drones::check_placements(world, shared, drones);
        }
        void free(){
            rlt::free(device, parameters);
            rlt::free(device, states);
            rlt::free(device, reset_mask);
            rlt::free(device, rng);
            rlt::free(device, world);
            rlt::free(device, shared.library);
        }
    };

}

using namespace test_drones;

// the rig coordinates are the rotor phases at shutter close, trailed at shutter open by the
// spin over the shutter fraction
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, ROTOR_COORDINATES_WITH_MOTION_BLUR){
    DEVICE device;
    BLUR_WORLD::Parameters parameters;
    BLUR_WORLD::DYNAMICS_ENV::State state{};
    parameters.shutter_fraction = (T)0.5;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        state.rotor_phase[rotor_i] = (T)0.3 * (T)(rotor_i + 1);
        state.rpm[rotor_i] = (T)60 * (T)(rotor_i + 1);
    }
    float open[4], close[4];
    rlt::rl::environments::hyperdrone::_drone_coordinates<DEVICE, BLUR_SPEC>(device, parameters, state, open, close);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        EXPECT_EQ(close[rotor_i], state.rotor_phase[rotor_i]);
        const T swept = (T)(rotor_i + 1) * (T)2 * rlt::math::PI<T> * parameters.dynamics.integration.dt * parameters.shutter_fraction;
        EXPECT_NEAR(open[rotor_i], state.rotor_phase[rotor_i] - swept, 1e-6);
    }
}

// a body swap rewrites only the agent regions' objects (in place, same slots and ids) and marks
// the overlay for upload only when the body changes; the props follow the entry's rig
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, BODY_SWAP_PRESERVES_WRAPPER_AND_AGENT_LAYOUT){
    DEVICE device;
    rlt::init(device);
    BLUR_WORLD world;
    BLUR_WORLD::SharedContext shared;
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto wrapper = rlt::add(device, shared.library.pool, quad("wrapper", 0, 0));
    const T rotor_positions[4][3] = {{0.1f, -0.1f, 0}, {-0.1f, 0.1f, 0}, {0.1f, 0.1f, 0}, {-0.1f, -0.1f, 0}};
    T rotor_torque_directions[4][3] = {{0, 0, 1}, {0, 0, 1}, {0, 0, -1}, {0, 0, -1}};
    rlt::rl::environments::hyperdrone::DroneBody<BLUR_SPEC> bodies[2];
    for(TI mesh_i = 0; mesh_i < 2; mesh_i++){
        const float reach = mesh_i == 0 ? 0.1f : 0.12f;
        rlt::rendering::raytracing::ObjectAssembly assembly;
        rlt::add(device, assembly, quad("body", 0, 0), identity);
        rlt::add(device, assembly, quad("prop_0", reach, -reach), identity);
        rlt::add(device, assembly, quad("prop_1", -reach, reach), identity);
        rlt::add(device, assembly, quad("prop_2", reach, reach), identity);
        rlt::add(device, assembly, quad("prop_3", -reach, -reach), identity);
        world.drone_meshes.push_back(rlt::add(device, shared.library.pool, assembly));
        rotor_torque_directions[0][2] = mesh_i == 0 ? (T)1 : (T)-1;
        bodies[mesh_i].mesh = mesh_i;
        ASSERT_TRUE(rlt::init(device, bodies[mesh_i].rig, assembly, rotor_positions, rotor_torque_directions, (TI)4));
    }
    BLUR_WORLD::EntityKind kind{wrapper, 0, {}};
    ASSERT_TRUE(rlt::init(device, kind.rig, shared.library.pool.assemblies[wrapper.index]));
    world.entity_kinds = {kind};
    world.drone_slot_offset = 1;
    world.drone_parts = 5;
    world.slots.resize(1);
    auto& renderer = rlt::get_renderer(device, world);
    rlt::rendering::Bundle<T> bundle;
    rlt::add(device, bundle.scene, quad("background", 0, 0), identity);
    rlt::rendering::datasets::compute_bounds(device, bundle);
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, bundle, shared.library.pool);
    const rlt::rendering::raytracing::OverlayIndex overlay{0};
    rlt::spawn(device, renderer, overlay, wrapper, identity);
    for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
        ASSERT_EQ(rlt::spawn(device, renderer, overlay, world.drone_meshes[0], identity).first_slot, 1 + agent_i * 5);
    }
    const auto wrapper_object = renderer.overlays[0].slots[0].object;
    BLUR_WORLD::Parameters parameters;
    parameters.shutter_fraction = 1;
    BLUR_WORLD::DYNAMICS_ENV::State state{};
    state.orientation[0] = 1;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        state.rotor_phase[rotor_i] = (T)0.3 * (T)(rotor_i + 1);
        state.rpm[rotor_i] = 600;
    }
    TI held = 0;
    for(TI mesh_i : {1, 0, 0, 1}){
        renderer.overlays[0].dirty = false;
        for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
            state.position[0] = (T)agent_i;
            float base[12];
            rlt::rl::environments::hyperdrone::rig::make_body_transform(device, state.orientation, state.position, parameters.scene_translation, parameters.scene_yaw_cos, parameters.scene_yaw_sin, base);
            float open[4], close[4];
            rlt::rl::environments::hyperdrone::_drone_coordinates<DEVICE, BLUR_SPEC>(device, parameters, state, open, close);
            rlt::rl::environments::hyperdrone::_drone_rows(device, world, renderer, (TI)0, agent_i, bodies[mesh_i], base, base, open, close);
        }
        EXPECT_EQ(renderer.overlays[0].dirty, mesh_i != held);
        held = mesh_i;
        EXPECT_EQ(renderer.overlays[0].slots[0].object, wrapper_object);
        const auto& record = renderer.assets[world.drone_meshes[mesh_i].index];
        for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
            const TI first_slot = 1 + agent_i * 5;
            for(TI part_i = 0; part_i < 5; part_i++){
                EXPECT_EQ(renderer.overlays[0].slots[first_slot + part_i].object, renderer.asset_part_objects[record.first_part + part_i]);
            }
            const TI prop_slot = first_slot + 1;
            EXPECT_NEAR(renderer.transforms_staging[prop_slot * 12], std::cos(state.rotor_phase[0]), 1e-6);
            const T spin = mesh_i == 0 ? (T)-1 : (T)1;
            EXPECT_NEAR(renderer.transforms_staging[prop_slot * 12 + 4], spin * std::sin(state.rotor_phase[0]), 1e-6);
            const TI last_sample = BLUR_SPEC::MOTION_BLUR_SAMPLES - 1;
            EXPECT_NE(renderer.transforms_motion_staging[prop_slot * 12], renderer.transforms_motion_staging[(last_sample * BLUR_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE + prop_slot) * 12]);
        }
        for(TI slot_i = 0; slot_i < BLUR_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE; slot_i++){
            EXPECT_EQ(renderer.overlays[0].slots[slot_i].active, slot_i < 1 + BLUR_SPEC::N_AGENTS * 5);
        }
    }
    rlt::free(device, renderer);
}

// a target setup carries its drone entry: the bank renders the entry's body at the target pose,
// a resetting instance flies the setup's entry, and the next frame swaps the body back
// the body at the target pose carries its mount to exactly the target camera
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, TARGET_BODY_POSE_CARRIES_THE_TARGET_CAMERA){
    DEVICE device;
    rlt::rl::environments::hyperdrone::Camera<T> camera;
    camera.mount.pose.position[0] = (T)0.1;
    camera.mount.pose.position[1] = (T)-0.05;
    camera.mount.pose.position[2] = (T)0.08;
    const T half_pitch = (T)-0.2;
    camera.mount.pose.orientation[0] = std::cos(half_pitch);
    camera.mount.pose.orientation[2] = std::sin(half_pitch);
    const T scene_translation[3] = {1, 2, 3};
    const T yaw = (T)0.7;
    for(T roll : {(T)0, (T)0.3, (T)-0.4}){
        for(T pitch : {(T)0, (T)0.25, (T)-0.15}){
            const auto target = rlt::rl::environments::hyperdrone::make_target_camera<DEVICE, T>(device, camera, (T)1.5, scene_translation, std::cos(yaw), std::sin(yaw), roll, pitch);
            T orientation[4], position[3];
            rlt::rl::environments::hyperdrone::make_target_body_pose(device, camera, roll, pitch, orientation, position);
            const auto carried = rlt::rl::environments::hyperdrone::make_camera<DEVICE, T>(device, camera, orientation, position, (T)1.5, scene_translation, std::cos(yaw), std::sin(yaw));
            for(TI axis = 0; axis < 3; axis++){
                EXPECT_NEAR(carried.pos[axis], target.pos[axis], 1e-5) << "roll " << roll << " pitch " << pitch;
                EXPECT_NEAR(carried.dir_00[axis], target.dir_00[axis], 1e-5) << "roll " << roll << " pitch " << pitch;
                EXPECT_NEAR(carried.dir_du[axis], target.dir_du[axis], 1e-5) << "roll " << roll << " pitch " << pitch;
                EXPECT_NEAR(carried.dir_dv[axis], target.dir_dv[axis], 1e-5) << "roll " << roll << " pitch " << pitch;
            }
        }
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, TARGET_FRAME_BANK_CARRIES_THE_DRONE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<WORLD>>;
    DEVICE device;
    rlt::init(device);
    TASK_WORLD world;
    TASK_WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    rlt::malloc(device, world);
    const auto table = one_asset_per_dynamics();
    shared.drones = table.drones;
    rlt::rendering::datasets::procthor::GLB dataset{{}, {SCENE_PATH}};
    typename decltype(dataset)::Corpus corpus;
    rlt::rendering::datasets::procthor::enumerate(device, dataset, corpus);
    rlt::init(device, world, shared, dataset, corpus, 0, 1, 0);
    InstanceTensor<TASK_WORLD::Parameters> parameters;
    InstanceTensor<TASK_WORLD::State> states;
    MASK_TENSOR mask;
    RNG rng;
    rlt::malloc(device, parameters);
    rlt::malloc(device, states);
    rlt::malloc(device, mask);
    rlt::malloc(device, rng);
    rlt::init(device, rng, 912);
    rlt::set_all(device, mask, true);
    rlt::sample_initial_parameters(device, world, parameters, mask, rng);
    rlt::sample_initial_state(device, world, parameters, states, mask, rng);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        rlt::set(device, mask, instance_i % 2 != 0, instance_i);
    }
    std::vector<TI> counts(table.drones.size(), 0);
    for(TI round_i = 0; round_i < 16; round_i++){
        rlt::select_scene(device, world, 0);
        std::vector<TASK_WORLD::Parameters> before(INSTANCES);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            before[instance_i] = rlt::get(device, parameters, instance_i);
        }
        rlt::sample_initial_parameters(device, world, parameters, mask, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const auto& setup = rlt::get_ref(device, world.setups, instance_i);
            ASSERT_LT(setup.drone, table.drones.size());
            EXPECT_EQ(setup.camera.fov, table.drones[setup.drone].parameters.camera.fov);
            EXPECT_EQ(rendered_mesh(world, 0, instance_i), shared.drone_bodies[setup.drone].mesh) << "the bank renders setup " << instance_i << "'s drone";
            const auto& actual = rlt::get_ref(device, parameters, instance_i);
            if(instance_i % 2 == 0){
                EXPECT_EQ(actual.drone, before[instance_i].drone);
                EXPECT_EQ(actual.dynamics.dynamics.mass, before[instance_i].dynamics.dynamics.mass);
                EXPECT_EQ(actual.camera.fov, before[instance_i].camera.fov);
            }
            else{
                const auto& adopted = rlt::get_ref(device, world.setups, rlt::get(device, world.setup_index, instance_i));
                EXPECT_EQ(actual.drone, adopted.drone);
                EXPECT_EQ(actual.camera.fov, adopted.camera.fov);
                EXPECT_EQ(actual.dynamics.dynamics.mass, table.drones[actual.drone].parameters.dynamics.dynamics.mass);
                counts[actual.drone]++;
            }
        }
        rlt::sample_initial_state(device, world, parameters, states, mask, rng);
        rlt::render(device, world, parameters, states, mask);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            EXPECT_EQ(rendered_mesh(world, 0, instance_i), shared.drone_bodies[rlt::get(device, parameters, instance_i).drone].mesh) << "instance " << instance_i;
        }
    }
    for(TI count : counts){
        EXPECT_GT(count, 0);
    }
    rlt::free(device, rng);
    rlt::free(device, mask);
    rlt::free(device, states);
    rlt::free(device, parameters);
    rlt::free(device, world);
    rlt::free(device, shared.library);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, TABLE_SHAPES){
    if(DATA_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    DEVICE device;
    rlt::init(device);
    const Table tables[] = {one_asset_two_dynamics(), one_asset_per_dynamics(), many_to_many()};
    for(const Table& table : tables){
        WORLD::SharedContext shared;
        rlt::malloc(device, shared.library);
        shared.drones = table.drones;
        rlt::init(device, shared);
        ASSERT_EQ(shared.num_drones, table.drones.size());
        TI distinct = 0;
        ASSERT_EQ(shared.drone_bodies.size(), table.drones.size());
        for(TI drone_i = 0; drone_i < (TI)table.drones.size(); drone_i++){
            EXPECT_EQ(shared.drone_bodies[drone_i].mesh, table.asset[drone_i]);
            EXPECT_EQ(shared.drone_bodies[drone_i].rig.num_coordinates, 4);
            distinct = table.asset[drone_i] + 1 > distinct ? table.asset[drone_i] + 1 : distinct;
        }
        ASSERT_EQ(shared.drone_meshes.size(), distinct);
        EXPECT_EQ(shared.library.pool.assemblies.size(), distinct);
        EXPECT_EQ(shared.drone_parts, 5);
        rlt::free(device, shared.library);
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, DEDUPLICATES_ASSEMBLY_CONTENT){
    if(DATA_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    const auto directory = std::filesystem::path(::testing::TempDir()) / "rl_tools_hyperdrone_drone_duplicate";
    std::filesystem::create_directories(directory);
    const auto copy_path = directory / "x500-copy.glb";
    auto table = one_asset_two_dynamics();
    ASSERT_TRUE(std::filesystem::copy_file(table.drones[0].asset, copy_path, std::filesystem::copy_options::overwrite_existing));
    table.drones[1].asset = copy_path.string();
    DEVICE device;
    rlt::init(device);
    WORLD::SharedContext shared;
    rlt::malloc(device, shared.library);
    shared.drones = table.drones;
    rlt::init(device, shared);
    std::filesystem::remove(copy_path);
    std::filesystem::remove(directory);
    EXPECT_NE(shared.drones[0].asset, shared.drones[1].asset);
    EXPECT_EQ(shared.num_drones, 2);
    EXPECT_EQ(shared.drone_bodies[0].mesh, 0);
    EXPECT_EQ(shared.drone_bodies[1].mesh, 0);
    EXPECT_EQ(shared.drone_meshes.size(), 1);
    EXPECT_EQ(shared.library.pool.assemblies.size(), 1);
    EXPECT_NE(shared.drones[0].parameters.camera.fov, shared.drones[1].parameters.camera.fov);
    rlt::free(device, shared.library);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, SAMPLING_COVERS_TABLE_AND_APPLIES_NOMINAL){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    {
        const Table table = many_to_many();
        Harness harness;
        harness.init(table.drones);
        std::vector<TI> counts(table.drones.size(), 0);
        for(TI round_i = 0; round_i < 64; round_i++){
            harness.reset();
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                const auto instance_parameters = rlt::get(harness.device, harness.parameters, instance_i);
                ASSERT_LT(instance_parameters.drone, table.drones.size());
                counts[instance_parameters.drone]++;
                const auto& nominal = table.drones[instance_parameters.drone].parameters;
                EXPECT_EQ(instance_parameters.dynamics.dynamics.mass, nominal.dynamics.dynamics.mass);
                EXPECT_EQ(instance_parameters.dynamics.dynamics.rotor_positions[1][1], nominal.dynamics.dynamics.rotor_positions[1][1]);
                EXPECT_EQ(instance_parameters.camera.fov, nominal.camera.fov);
            }
        }
        for(TI drone_i = 0; drone_i < table.drones.size(); drone_i++){
            EXPECT_GT(counts[drone_i], 0) << "drone " << drone_i << " never sampled";
        }
        harness.free();
    }
    {
        Harness harness;
        harness.init({drone<REGISTRY::arpl>("arpl.glb", 70)});
        for(TI round_i = 0; round_i < 8; round_i++){
            harness.reset();
            for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
                const auto instance_parameters = rlt::get(harness.device, harness.parameters, instance_i);
                EXPECT_EQ(instance_parameters.drone, 0);
                EXPECT_EQ(instance_parameters.camera.fov, 70);
                EXPECT_EQ(instance_parameters.dynamics.dynamics.mass, (l2f::parameters::dynamics::registry<REGISTRY::arpl, PARAMETERS_SPEC>).mass);
            }
        }
        harness.free();
    }
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, RENDER_SWAPS_BODIES_PER_INSTANCE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Harness harness;
    const auto table = one_asset_per_dynamics();
    harness.init(table.drones);
    EXPECT_EQ(harness.world.num_drones, table.drones.size());
    EXPECT_EQ(harness.world.dynamics.parameters.dynamics.mass, table.drones[0].parameters.dynamics.dynamics.mass);
    EXPECT_EQ(harness.world.parameters.camera.fov, table.drones[0].parameters.camera.fov);
    EXPECT_TRUE(harness.world.entity_kinds.empty());
    harness.check_placements({0, 0, 0, 0});
    harness.reset();
    // one assembly per instance
    harness.arrange({0, 1, 2, 3});
    harness.render(true);
    harness.check_placements({0, 1, 2, 3});
    const auto masks_distinct = harness.own_masks();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_TRUE(seen(masks_distinct[instance_i])) << "camera " << instance_i << " should see its own drone";
        for(TI other_i = instance_i + 1; other_i < INSTANCES; other_i++){
            EXPECT_NE(masks_distinct[instance_i], masks_distinct[other_i]) << "assemblies " << instance_i << " and " << other_i << " should render differently";
        }
    }
    // every instance onto the x500: same assembly at the same relative pose, same mask, the
    // same slots whatever was rendered before
    harness.arrange({3, 3, 3, 3});
    harness.render(false);
    harness.check_placements({3, 3, 3, 3});
    const auto masks_x500 = harness.own_masks();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(masks_x500[instance_i], masks_distinct[3]) << "instance " << instance_i;
    }
    // and back: the swap leaves no trace
    harness.arrange({0, 1, 2, 3});
    harness.render(false);
    harness.check_placements({0, 1, 2, 3});
    const auto masks_back = harness.own_masks();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(masks_back[instance_i], masks_distinct[instance_i]) << "instance " << instance_i;
    }
    harness.free();
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, ROTATE_SCENE_SWAPS_PER_SLOT){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Harness harness;
    harness.init(one_asset_per_dynamics().drones, 2);
    ASSERT_EQ(harness.world.slots.size(), 2);
    harness.reset();
    harness.arrange({0, 1, 2, 3});
    harness.render(true);
    harness.check_placements({0, 1, 2, 3});
    const auto masks_first = harness.own_masks();
    rlt::rotate_scene(harness.device, harness.world);
    EXPECT_EQ(harness.world.active_slot, 1);
    harness.reset();
    harness.arrange({3, 2, 1, 0});
    harness.render(true);
    harness.check_placements({3, 2, 1, 0});
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(rendered_mesh(harness.world, 0, instance_i), instance_i) << "the inactive slot keeps its bodies";
    }
    rlt::rotate_scene(harness.device, harness.world);
    EXPECT_EQ(harness.world.active_slot, 0);
    // back on the first slot with its bodies
    harness.reset();
    harness.arrange({0, 1, 2, 3});
    harness.render(true);
    harness.check_placements({0, 1, 2, 3});
    const auto masks_again = harness.own_masks();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(masks_again[instance_i], masks_first[instance_i]) << "instance " << instance_i;
    }
    harness.reset();
    harness.arrange({1, 2, 3, 0});
    harness.render(true);
    harness.check_placements({1, 2, 3, 0});
    const auto masks = harness.own_masks();
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_TRUE(seen(masks[instance_i])) << "camera " << instance_i;
        EXPECT_EQ(masks[instance_i], masks_first[(instance_i + 1) % INSTANCES]) << "instance " << instance_i;
    }
    harness.free();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
