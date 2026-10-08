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
            test_drones::arrange(device, parameters, states, drones);
        }
        void render(bool reset){
            rlt::set_all(device, reset_mask, reset);
            rlt::render(device, world, parameters, states, reset_mask);
        }
        std::vector<std::vector<uint32_t>> own_masks(){
            return test_drones::own_masks(device, device, world, shared);
        }
        void check_placements(const std::vector<TI>& drones){
            test_drones::check_placements(world, drones);
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, EFFECTIVE_ROTOR_PHASES_WITH_MOTION_BLUR){
    DEVICE device;
    BLUR_WORLD::Parameters parameters;
    BLUR_WORLD::DYNAMICS_ENV::State state{};
    state.orientation[0] = 1;
    parameters.drone = 3;
    parameters.shutter_fraction = (T)0.5;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        state.rotor_phase[rotor_i] = (T)0.3 * (T)(rotor_i + 1);
        state.rpm[rotor_i] = (T)60 * (T)(rotor_i + 1);
    }
    rlt::rl::environments::hyperdrone::DronePose<BLUR_SPEC> pose;
    rlt::make_drone_pose(device, parameters, state, pose);
    ASSERT_TRUE(pose.valid);
    EXPECT_EQ(pose.drone, 3);
    const TI quadrant_to_rotor[4] = {1, 0, 2, 3};
    for(TI quadrant = 0; quadrant < 4; quadrant++){
        const TI rotor_i = quadrant_to_rotor[quadrant];
        EXPECT_EQ(pose.phase_close[quadrant], state.rotor_phase[rotor_i]);
        const T swept = (T)(rotor_i + 1) * (T)2 * rlt::math::PI<T> * parameters.dynamics.integration.dt * parameters.shutter_fraction;
        EXPECT_NEAR(pose.phase_open[quadrant], state.rotor_phase[rotor_i] - swept, 1e-6);
    }
    for(TI axis = 0; axis < 3; axis++){
        std::swap(parameters.dynamics.dynamics.rotor_positions[0][axis], parameters.dynamics.dynamics.rotor_positions[2][axis]);
    }
    rlt::make_drone_pose(device, parameters, state, pose);
    ASSERT_TRUE(pose.valid);
    EXPECT_EQ(pose.phase_close[1], state.rotor_phase[2]);
    EXPECT_EQ(pose.phase_close[2], state.rotor_phase[0]);
    parameters.dynamics.dynamics.rotor_positions[0][0] = 0;
    rlt::make_drone_pose(device, parameters, state, pose);
    EXPECT_FALSE(pose.valid);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, VARIABLE_ASSEMBLIES_PRESERVE_WRAPPER_AND_AGENT_LAYOUT){
    DEVICE device;
    rlt::init(device);
    BLUR_WORLD world;
    BLUR_WORLD::SharedContext shared;
    world.shared = &shared;
    world.num_drones = 2;
    shared.num_drones = 2;
    shared.drone_asset = {0, 1};
    const float identity[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    const auto wrapper = rlt::add(device, shared.library.pool, quad("wrapper", 0, 0));
    for(TI asset_i = 0; asset_i < 2; asset_i++){
        rlt::rendering::raytracing::ObjectAssembly assembly;
        rlt::add(device, assembly, quad("body", 0, 0), identity);
        rlt::add(device, assembly, quad("prop_0", 0.1f, -0.1f), identity);
        rlt::add(device, assembly, quad("prop_1", -0.1f, 0.1f), identity);
        rlt::add(device, assembly, quad("prop_2", 0.1f, 0.1f), identity);
        rlt::add(device, assembly, quad("prop_3", -0.1f, -0.1f), identity);
        if(asset_i == 1){
            rlt::add(device, assembly, quad("antenna", 0, 0.2f), identity);
        }
        rlt::rl::environments::hyperdrone::DroneAsset<BLUR_SPEC> asset;
        asset.asset = rlt::add(device, shared.library.pool, assembly);
        asset.parts = (TI)assembly.parts.size();
        ASSERT_TRUE(rlt::init(device, asset.rig, assembly));
        const TI prop_quadrants[4] = {1, 2, 3, 0};
        for(TI prop_i = 0; prop_i < 4; prop_i++){
            asset.prop_quadrants[prop_i] = prop_quadrants[prop_i];
        }
        shared.drone_assets.push_back(asset);
    }
    world.drone_entity_kind_index = 1;
    world.entity_kinds = {{wrapper, 1, 0}, {shared.drone_assets[0].asset, 5, 0}, {shared.drone_assets[0].asset, 5, 0}};
    world.slots.resize(1);
    auto& slot = world.slots[0];
    auto& renderer = rlt::get_renderer(device, world);
    rlt::rendering::Bundle<T> bundle;
    rlt::add(device, bundle.scene, quad("background", 0, 0), identity);
    rlt::rendering::datasets::compute_bounds(device, bundle);
    rlt::malloc(device, renderer);
    rlt::init(device, renderer, bundle, shared.library.pool);
    for(const auto& kind : world.entity_kinds){
        slot.entity_placements.push_back(rlt::spawn(device, renderer, rlt::rendering::raytracing::OverlayIndex{0}, kind.asset, identity));
    }
    slot.drone_spawned_asset = {0};
    const auto wrapper_object = renderer.overlays[0].slots[0].object;
    BLUR_WORLD::Parameters parameters;
    parameters.shutter_fraction = 1;
    BLUR_WORLD::DYNAMICS_ENV::State state{};
    state.orientation[0] = 1;
    for(TI rotor_i = 0; rotor_i < 4; rotor_i++){
        state.rotor_phase[rotor_i] = (T)0.3 * (T)(rotor_i + 1);
        state.rpm[rotor_i] = 600;
    }
    rlt::rl::environments::hyperdrone::DronePose<BLUR_SPEC> poses[BLUR_SPEC::N_AGENTS];
    for(TI asset_i : {1, 0, 1, 0}){
        parameters.drone = asset_i;
        for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
            state.position[0] = (T)agent_i;
            rlt::make_drone_pose(device, parameters, state, poses[agent_i]);
        }
        rlt::apply_drone_poses(device, world, (TI)0, poses);
        const TI parts = shared.drone_assets[asset_i].parts;
        EXPECT_EQ(slot.drone_spawned_asset[0], asset_i);
        EXPECT_EQ(renderer.overlays[0].slots[0].object, wrapper_object);
        EXPECT_TRUE(renderer.overlays[0].slots[0].active);
        for(TI element = 0; element < 12; element++){
            EXPECT_EQ(renderer.transforms_staging[element], identity[element]);
        }
        for(TI agent_i = 0; agent_i < BLUR_SPEC::N_AGENTS; agent_i++){
            const auto& placement = slot.entity_placements[1 + agent_i];
            ASSERT_EQ(placement.first_slot, 1 + agent_i * parts);
            ASSERT_EQ(placement.num_parts, parts);
            const TI prop_slot = (TI)placement.first_slot + 1;
            EXPECT_NEAR(renderer.transforms_staging[prop_slot * 12], std::cos(state.rotor_phase[0]), 1e-6);
            const TI last_sample = BLUR_SPEC::MOTION_BLUR_SAMPLES - 1;
            EXPECT_NE(renderer.transforms_motion_staging[prop_slot * 12], renderer.transforms_motion_staging[(last_sample * BLUR_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE + prop_slot) * 12]);
        }
        for(TI slot_i = 0; slot_i < BLUR_SPEC::MAX_ENTITY_SLOTS_PER_INSTANCE; slot_i++){
            EXPECT_EQ(renderer.overlays[0].slots[slot_i].active, slot_i < 1 + BLUR_SPEC::N_AGENTS * parts);
        }
    }
    rlt::free(device, renderer);
}

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, TARGET_FRAME_PARTIAL_RESET){
    using TASK_SPEC = rlt::rl::environments::hyperdrone::tasks::target_frame::Specification<SAMPLING_WORLD>;
    using TASK_WORLD = rlt::rl::environments::hyperdrone::tasks::target_frame::World<TASK_SPEC>;
    DEVICE device;
    rlt::init(device);
    TASK_WORLD world;
    const auto drones = one_asset_per_dynamics().drones;
    init_sampling_table(device, device, world, drones);
    InstanceTensor<TASK_WORLD::Parameters> parameters;
    MASK_TENSOR mask;
    RNG rng, reference_rng;
    rlt::malloc(device, parameters);
    rlt::malloc(device, mask);
    rlt::malloc(device, rng);
    rlt::malloc(device, reference_rng);
    rlt::init(device, rng, 912);
    rlt::init(device, reference_rng, 912);
    TASK_WORLD::Parameters expected[INSTANCES];
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        expected[instance_i].drone = instance_i;
        expected[instance_i].dynamics.dynamics.mass = (T)(10 + instance_i);
        expected[instance_i].camera.fov = (T)(100 + instance_i);
        expected[instance_i].target_pose[0] = (T)(9 + instance_i);
        rlt::set(device, parameters, expected[instance_i], instance_i);
        rlt::set(device, mask, instance_i % 2 != 0, instance_i);
    }
    std::vector<TI> counts(drones.size(), 0);
    for(TI round_i = 0; round_i < 32; round_i++){
        rlt::sample_initial_parameters(device, world, parameters, mask, rng);
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            if(instance_i % 2 != 0){
                rlt::sample_initial_parameters(device, world, expected[instance_i], reference_rng);
            }
            const auto& actual = rlt::get_ref(device, parameters, instance_i);
            EXPECT_EQ(actual.drone, expected[instance_i].drone);
            EXPECT_EQ(actual.dynamics.dynamics.mass, expected[instance_i].dynamics.dynamics.mass);
            EXPECT_EQ(actual.camera.fov, expected[instance_i].camera.fov);
            EXPECT_EQ(actual.shutter_fraction, expected[instance_i].shutter_fraction);
            EXPECT_EQ(actual.target_pose[0], expected[instance_i].target_pose[0]);
            if(instance_i % 2 != 0){
                ASSERT_LT(actual.drone, drones.size());
                const auto& nominal = drones[actual.drone].parameters;
                EXPECT_EQ(actual.dynamics.dynamics.mass, nominal.dynamics.dynamics.mass);
                EXPECT_EQ(actual.camera.fov, nominal.camera.fov);
                counts[actual.drone]++;
            }
        }
    }
    for(TI count : counts){
        EXPECT_GT(count, 0);
    }
    rlt::free(device, reference_rng);
    rlt::free(device, rng);
    rlt::free(device, mask);
    rlt::free(device, parameters);
    rlt::free(device, world.drone_parameters);
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
        ASSERT_EQ(shared.drone_asset, table.asset);
        TI distinct = 0;
        for(TI asset_i : table.asset){
            distinct = asset_i + 1 > distinct ? asset_i + 1 : distinct;
        }
        ASSERT_EQ(shared.drone_assets.size(), distinct);
        EXPECT_EQ(shared.library.pool.assemblies.size(), distinct);
        for(const auto& asset : shared.drone_assets){
            EXPECT_EQ(asset.parts, 5);
            EXPECT_EQ(asset.rig.num_coordinates, 4);
            unsigned quadrants = 0;
            for(TI quadrant : asset.prop_quadrants){
                ASSERT_LT(quadrant, 4);
                quadrants |= 1u << quadrant;
            }
            EXPECT_EQ(quadrants, 15u);
        }
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
    EXPECT_EQ(shared.drone_asset, (std::vector<TI>{0, 0}));
    EXPECT_EQ(shared.drone_assets.size(), 1);
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, RENDER_RECONCILES_PER_INSTANCE){
    if(SCENE_PATH.empty()){
        GTEST_SKIP() << "RL_TOOLS_TEST_DATA_PATH not set";
    }
    Harness harness;
    const auto table = one_asset_per_dynamics();
    harness.init(table.drones);
    EXPECT_EQ(harness.world.shared, &harness.shared);
    EXPECT_EQ(harness.world.num_drones, table.drones.size());
    EXPECT_EQ(harness.world.dynamics.parameters.dynamics.mass, table.drones[0].parameters.dynamics.dynamics.mass);
    EXPECT_EQ(harness.world.parameters.camera.fov, table.drones[0].parameters.camera.fov);
    EXPECT_EQ(harness.world.entity_kinds.size(), 1);
    ASSERT_EQ(harness.world.slots[0].entity_placements.size(), INSTANCES);
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        EXPECT_EQ(harness.world.slots[0].drone_spawned_asset[instance_i], 0);
    }
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
    // every instance onto the x500: same assembly at the same relative pose, same mask, ids
    // packed from slot 0 whatever was spawned before
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

TEST(RL_TOOLS_RL_ENVIRONMENTS_HYPERDRONE_DRONES, ROTATE_SCENE_RECONCILES){
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
        EXPECT_EQ(harness.world.slots[0].drone_spawned_asset[instance_i], instance_i) << "the inactive slot keeps its record";
    }
    rlt::rotate_scene(harness.device, harness.world);
    EXPECT_EQ(harness.world.active_slot, 0);
    // back on the first slot with its recorded assemblies: the view must still hold them
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
