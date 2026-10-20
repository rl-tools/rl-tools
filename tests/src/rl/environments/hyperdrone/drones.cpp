#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/tasks/target_frame/operations_cpu.h>

#include "../../../utils/utils.h"

#include "drones_test.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <utility>

namespace test_drones {
    struct BLUR_SPEC: WORLD_SPEC {
        static constexpr TI INSTANCES_PER_WORLD = 1;
        static constexpr TI N_AGENTS = 2;
        static constexpr TI N_CAMERA_AGENTS = 2;
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
            rlt::request_render(device, world);
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

    template <typename TEST_WORLD>
    float rendered_rotor_span(const TEST_WORLD& world, const typename TEST_WORLD::SharedContext& shared, TI instance_i, TI drone_i){
        const auto& renderer = world.slots[world.active_slot].renderer;
        const auto& body = shared.drone_bodies[drone_i];
        const auto& assembly = shared.library.pool.assemblies[world.drone_meshes[body.mesh].index];
        std::vector<std::array<float, 3>> hubs;
        for(TI part_i = 0; part_i < assembly.parts.size(); part_i++){
            const auto& object = assembly.objects[assembly.parts[part_i].object];
            if(object.name.rfind("prop_", 0) != 0){
                continue;
            }
            const float limit = std::numeric_limits<float>::max();
            float low[2] = {limit, limit}, high[2] = {-limit, -limit};
            for(const auto& mesh : object.meshes){
                for(TI vertex_i = 0; vertex_i < mesh.vertices.size(); vertex_i += 3){
                    for(TI axis = 0; axis < 2; axis++){
                        low[axis] = std::min(low[axis], mesh.vertices[vertex_i + axis]);
                        high[axis] = std::max(high[axis], mesh.vertices[vertex_i + axis]);
                    }
                }
            }
            const float pivot[3] = {(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, 0};
            std::array<float, 3> hub;
            const TI slot_i = instance_i * TEST_WORLD::SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE + world.drone_slot_offset + part_i;
            rlt::rendering::transform_point(renderer.transforms_staging.data() + slot_i * 12, pivot, hub.data());
            hubs.push_back(hub);
        }
        EXPECT_EQ(hubs.size(), 4);
        float span_squared = 0;
        for(TI first = 0; first < hubs.size(); first++){
            for(TI second = first + 1; second < hubs.size(); second++){
                float distance_squared = 0;
                for(TI axis = 0; axis < 3; axis++){
                    const float delta = hubs[first][axis] - hubs[second][axis];
                    distance_squared += delta * delta;
                }
                span_squared = std::max(span_squared, distance_squared);
            }
        }
        return std::sqrt(span_squared);
    }
}

using namespace test_drones;

// the rig coordinates are the rotor phases at shutter close, trailed at shutter open by the
// spin over the shutter fraction
TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, ROTOR_COORDINATES_WITH_MOTION_BLUR){
    DEVICE device;
    BLUR_WORLD::Parameters parameters;
    BLUR_WORLD::DYNAMICS_ENV::State state{};
    parameters.shutter_fraction = (T)0.5;
    parameters.agents[0].dynamics.dynamics.max_rpm = parameters.agents[0].dynamics.dynamics.action_limit.max; // identity calibration: the motor state reads as RPM
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        state.rotor_phase[rotor_i] = (T)0.3 * (T)(rotor_i + 1);
        state.rpm[rotor_i] = (T)60 * (T)(rotor_i + 1);
    }
    rlt::rl::environments::hyperdrone::Rotorcraft identity;
    identity.num_props = 4;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        identity.prop_rotor[rotor_i] = rotor_i;
        identity.prop_spin[rotor_i] = 1;
    }
    float open[4], close[4];
    rlt::rl::environments::hyperdrone::_drone_coordinates<DEVICE, BLUR_SPEC>(device, parameters, (TI)0, state, identity, open, close);
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        EXPECT_EQ(close[rotor_i], state.rotor_phase[rotor_i]);
        const T swept = (T)(rotor_i + 1) * (T)2 * rlt::math::PI<T> * parameters.agents[0].dynamics.integration.dt * parameters.shutter_fraction;
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
        rotor_torque_directions[0][2] = mesh_i == 0 ? (T)1 : (T)-1;
        bodies[mesh_i].mesh = mesh_i;
        ASSERT_TRUE(rlt::articulate(device, assembly));
        ASSERT_TRUE(rlt::init(device, bodies[mesh_i].rotorcraft, assembly, rotor_positions, rotor_torque_directions, (TI)4));
        world.drone_meshes.push_back(rlt::add(device, shared.library.pool, assembly));
    }
    world.entity_kinds = {BLUR_WORLD::EntityKind{wrapper, 0, 1}};
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
    rlt::spawn(device, renderer, overlay, wrapper);
    for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
        ASSERT_EQ(rlt::spawn(device, renderer, overlay, world.drone_meshes[0]).first_slot, 1 + agent_i * 5);
    }
    const auto wrapper_object = renderer.overlays[0].slots[0].object;
    BLUR_WORLD::Parameters parameters;
    parameters.shutter_fraction = 1;
    for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
        parameters.agents[agent_i].dynamics.dynamics.max_rpm = parameters.agents[agent_i].dynamics.dynamics.action_limit.max; // identity calibration: the motor state reads as RPM
    }
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
            rlt::rl::environments::hyperdrone::make_body_transform(device, state.orientation, state.position, parameters.scene_translation, parameters.scene_yaw_cos, parameters.scene_yaw_sin, base);
            float open[4], close[4];
            rlt::rl::environments::hyperdrone::_drone_coordinates<DEVICE, BLUR_SPEC>(device, parameters, agent_i, state, bodies[mesh_i].rotorcraft, open, close);
            const float scale = bodies[mesh_i].rotorcraft.scale;
            rlt::rl::environments::hyperdrone::_drone_rows(device, world, renderer, (TI)0, agent_i, bodies[mesh_i], rlt::make_pose(base, scale), open, close);
        }
        EXPECT_EQ(renderer.overlays[0].dirty, mesh_i != held);
        held = mesh_i;
        EXPECT_EQ(renderer.overlays[0].slots[0].object, wrapper_object);
        const auto& record = renderer.assets[world.drone_meshes[mesh_i].index];
        for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
            const TI first_slot = 1 + agent_i * 5;
            for(TI part_i = 0; part_i < 5; part_i++){
                EXPECT_EQ(renderer.overlays[0].slots[first_slot + part_i].object, renderer.asset_parts[record.first_part + part_i].object);
            }
            const TI prop_slot = first_slot + 1;
            const T scale = mesh_i == 0 ? 1.0f : 0.1f / 0.12f;
            EXPECT_NEAR(renderer.transforms_staging[prop_slot * 12], scale * std::cos(state.rotor_phase[0]), 1e-6);
            const T spin = mesh_i == 0 ? (T)-1 : (T)1;
            EXPECT_NEAR(renderer.transforms_staging[prop_slot * 12 + 4], scale * spin * std::sin(state.rotor_phase[0]), 1e-6);
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
            ASSERT_LT(setup.agents[0].drone, table.drones.size());
            EXPECT_EQ(setup.agents[0].cameras[0].fov, table.drones[setup.agents[0].drone].parameters.cameras[0].fov);
            EXPECT_EQ(rendered_mesh(world, 0, instance_i), shared.drone_bodies[setup.agents[0].drone].mesh) << "the bank renders setup " << instance_i << "'s drone";
            const auto& actual = rlt::get_ref(device, parameters, instance_i);
            if(instance_i % 2 == 0){
                EXPECT_EQ(actual.agents[0].drone, before[instance_i].agents[0].drone);
                EXPECT_EQ(actual.agents[0].dynamics.dynamics.mass, before[instance_i].agents[0].dynamics.dynamics.mass);
                EXPECT_EQ(actual.agents[0].cameras[0].fov, before[instance_i].agents[0].cameras[0].fov);
            }
            else{
                const auto& adopted = rlt::get_ref(device, world.setups, rlt::get(device, world.setup_index, instance_i));
                EXPECT_EQ(actual.agents[0].drone, adopted.agents[0].drone);
                EXPECT_EQ(actual.agents[0].cameras[0].fov, adopted.agents[0].cameras[0].fov);
                EXPECT_EQ(actual.agents[0].dynamics.dynamics.mass, table.drones[actual.agents[0].drone].parameters.dynamics.dynamics.mass);
                counts[actual.agents[0].drone]++;
            }
        }
        // re-rendering the reset instances' targets from their adopted parameters reproduces the
        // bank frames they copied and leaves the others alone
        std::vector<float> copied(INSTANCES * WORLD::FRAME_DIM);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI element_i = 0; element_i < WORLD::FRAME_DIM; element_i++){
                copied[instance_i * WORLD::FRAME_DIM + element_i] = rlt::get(device, world.target_frames, instance_i, element_i);
            }
        }
        rlt::render_target_frames(device, world, parameters, mask);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI element_i = 0; element_i < WORLD::FRAME_DIM; element_i++){
                ASSERT_EQ(rlt::get(device, world.target_frames, instance_i, element_i), copied[instance_i * WORLD::FRAME_DIM + element_i]) << "instance " << instance_i << " element " << element_i;
            }
        }
        rlt::sample_initial_state(device, world, parameters, states, mask, rng);
        rlt::render(device, world, parameters, states, mask);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            EXPECT_EQ(rendered_mesh(world, 0, instance_i), shared.drone_bodies[rlt::get(device, parameters, instance_i).agents[0].drone].mesh) << "instance " << instance_i;
        }
    }
    for(TI count : counts){
        EXPECT_GT(count, 0);
    }
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        auto& setup = rlt::get_ref(device, world.setups, instance_i);
        setup.agents[0].drone = 0;
        setup.agents[0].dynamics.dynamics = table.drones[0].parameters.dynamics.dynamics;
        const T factor = T(instance_i + 1) / 2;
        for(auto& rotor : setup.agents[0].dynamics.dynamics.rotor_positions){
            for(auto& coordinate : rotor){
                coordinate *= factor;
            }
        }
        setup.target_roll = T(0.2);
        setup.target_pitch = T(-0.3);
    }
    rlt::render_target_bank(device, world);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        const T factor = T(instance_i + 1) / 2;
        EXPECT_NEAR(rendered_rotor_span(world, shared, instance_i, 0), factor * std::sqrt(2.0f) * 0.056f, 2e-5f);
        auto& instance_parameters = rlt::get_ref(device, parameters, instance_i);
        instance_parameters = rlt::get_ref(device, world.setups, instance_i);
        for(auto& rotor : instance_parameters.agents[0].dynamics.dynamics.rotor_positions){
            for(auto& coordinate : rotor){
                coordinate *= T(0.75);
            }
        }
    }
    rlt::set_all(device, mask, true);
    rlt::render_target_frames(device, world, parameters, mask);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        const T factor = T(instance_i + 1) / 2;
        EXPECT_NEAR(rendered_rotor_span(world, shared, instance_i, 0), factor * 0.75f * std::sqrt(2.0f) * 0.056f, 2e-5f);
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
            EXPECT_EQ(rlt::num_coordinates(device, shared.library.pool.assemblies[shared.drone_meshes[shared.drone_bodies[drone_i].mesh].index]), 4);
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
    EXPECT_NE(shared.drones[0].parameters.cameras[0].fov, shared.drones[1].parameters.cameras[0].fov);
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
                ASSERT_LT(instance_parameters.agents[0].drone, table.drones.size());
                counts[instance_parameters.agents[0].drone]++;
                const auto& nominal = table.drones[instance_parameters.agents[0].drone].parameters;
                EXPECT_EQ(instance_parameters.agents[0].dynamics.dynamics.mass, nominal.dynamics.dynamics.mass);
                EXPECT_EQ(instance_parameters.agents[0].dynamics.dynamics.rotor_positions[1][1], nominal.dynamics.dynamics.rotor_positions[1][1]);
                EXPECT_EQ(instance_parameters.agents[0].cameras[0].fov, nominal.cameras[0].fov);
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
                EXPECT_EQ(instance_parameters.agents[0].drone, 0);
                EXPECT_EQ(instance_parameters.agents[0].cameras[0].fov, 70);
                EXPECT_EQ(instance_parameters.agents[0].dynamics.dynamics.mass, (l2f::parameters::dynamics::registry<REGISTRY::arpl, PARAMETERS_SPEC>).mass);
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
    EXPECT_EQ(harness.world.parameters.agents[0].cameras[0].fov, table.drones[0].parameters.cameras[0].fov);
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, CURRENT_WHEELBASE_SCALES_EACH_INSTANCE_WITH_SHARED_GEOMETRY){
    using rlt::get_ref;
    using rlt::request_render;
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Harness harness;
    harness.init({drone<REGISTRY::crazyflie>("crazyflie.glb", 60)});
    harness.reset();
    harness.arrange({0,0,0,0});
    const auto original_rotorcraft = get_ref(harness.device, harness.world.drone_bodies, TI(0)).rotorcraft;
    const auto original_assembly = harness.shared.library.pool.assemblies.front();
    for(TI iteration = 0; iteration < 2; iteration++){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            auto& parameters = get_ref(harness.device, harness.parameters, instance_i);
            auto& state = get_ref(harness.device, harness.states, instance_i);
            parameters.agents[0].dynamics.dynamics = harness.shared.drones[0].parameters.dynamics.dynamics;
            const T factor = T(iteration == 0 ? instance_i + 1 : INSTANCES - instance_i) / 2;
            for(auto& rotor : parameters.agents[0].dynamics.dynamics.rotor_positions){
                for(auto& coordinate : rotor){
                    coordinate *= factor;
                }
            }
            parameters.scene_yaw_cos = 0;
            parameters.scene_yaw_sin = 1;
            state.orientation[0] = std::cos(T(0.2));
            state.orientation[1] = std::sin(T(0.2));
            state.position[0] = T(0.5);
            state.position[1] = T(-0.25);
            state.position[2] = T(0.2);
            for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
                state.rotor_phase[rotor_i] = T(iteration + 1) * T(rotor_i + 1) * T(0.3);
            }
        }
        request_render(harness.device, harness.world);
        harness.render(iteration == 0);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            const T factor = T(iteration == 0 ? instance_i + 1 : INSTANCES - instance_i) / 2;
            EXPECT_NEAR(rendered_rotor_span(harness.world, harness.shared, instance_i, 0), factor * std::sqrt(2.0f) * 0.056f, 2e-5f);
        }
        harness.check_placements({0,0,0,0});
    }
    EXPECT_EQ(harness.shared.drone_meshes.size(), 1);
    EXPECT_EQ(harness.shared.library.pool.assemblies.size(), 1);
    const auto& assembly = harness.shared.library.pool.assemblies.front();
    for(TI object_i = 0; object_i < assembly.objects.size(); object_i++){
        for(TI mesh_i = 0; mesh_i < assembly.objects[object_i].meshes.size(); mesh_i++){
            EXPECT_EQ(assembly.objects[object_i].meshes[mesh_i].vertices, original_assembly.objects[object_i].meshes[mesh_i].vertices);
        }
    }
    const auto& current_rotorcraft = get_ref(harness.device, harness.world.drone_bodies, TI(0)).rotorcraft;
    EXPECT_EQ(current_rotorcraft.wheelbase, original_rotorcraft.wheelbase);
    EXPECT_EQ(current_rotorcraft.scale, original_rotorcraft.scale);
    for(TI part_i = 0; part_i < assembly.parts.size(); part_i++){
        const auto& joint = assembly.parts[part_i].joint;
        const auto& original = original_assembly.parts[part_i].joint;
        EXPECT_TRUE(joint.type == original.type && joint.coordinate == original.coordinate);
        for(TI dim = 0; dim < 3; dim++){
            EXPECT_EQ(joint.axis[dim], original.axis[dim]);
            EXPECT_EQ(joint.pivot[dim], original.pivot[dim]);
        }
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
