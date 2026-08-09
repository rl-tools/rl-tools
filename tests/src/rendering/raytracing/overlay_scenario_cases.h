#ifndef TESTS_RENDERING_RAYTRACING_OVERLAY_SCENARIO_CASES_H
#define TESTS_RENDERING_RAYTRACING_OVERLAY_SCENARIO_CASES_H

#include <rl_tools/operations/cpu.h>
#include <rl_tools/rendering/raytracing/operations_cpu_common.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace overlay_scenarios {
    static constexpr std::size_t NUM_CAMERAS = 4;
    static constexpr std::size_t NUM_OVERLAYS = 10;
    static constexpr std::size_t MAX_OVERLAY_INSTANCES = 4;
    static constexpr std::size_t MAX_OVERLAYS_PER_CAMERA = 4;

    enum class Scenario {
        SHARED_SCENE_NO_DYNAMIC,
        ALL_SHARED_MESH_TRANSFORM,
        PARTIALLY_SHARED_MESH_TRANSFORM,
        SHARED_MESH_INDIVIDUAL_TRANSFORM,
        DISJOINT,
        MIXED,
    };

    inline constexpr std::array<Scenario, 6> SCENARIOS = {{
        Scenario::SHARED_SCENE_NO_DYNAMIC,
        Scenario::ALL_SHARED_MESH_TRANSFORM,
        Scenario::PARTIALLY_SHARED_MESH_TRANSFORM,
        Scenario::SHARED_MESH_INDIVIDUAL_TRANSFORM,
        Scenario::DISJOINT,
        Scenario::MIXED,
    }};

    using Transform = std::array<float, 12>;
    using CameraMask = std::uint8_t;

    constexpr CameraMask camera_bit(std::size_t camera){
        return static_cast<CameraMask>(1u << camera);
    }

    static constexpr CameraMask ALL_CAMERAS = static_cast<CameraMask>((1u << NUM_CAMERAS) - 1u);

    struct AssetDefinition {
        const char* name;
        std::array<float, 3> color;
        float half_extent;
        int dominant_channel;
    };

    struct PlacementDefinition {
        std::size_t overlay;
        std::size_t asset;
        CameraMask cameras;
        Transform initial_transform;
        Transform updated_transform;
        bool moves_in_updated_state;
        std::size_t expected_first_slot;
        std::uint32_t expected_id;
    };

    struct Definition {
        Scenario scenario;
        const char* id;
        std::vector<AssetDefinition> assets;
        std::vector<PlacementDefinition> placements;
        CameraMask affected_camera_mask;
    };

    constexpr Transform pose(float x, float y, float z){
        return {{1, 0, 0, x, 0, 1, 0, y, 0, 0, 1, z}};
    }

    constexpr float camera_y(std::size_t camera){
        return -1.2f + 0.8f * static_cast<float>(camera);
    }

    inline const Definition& definition(Scenario scenario){
        static const Definition shared_scene_no_dynamic = {
            Scenario::SHARED_SCENE_NO_DYNAMIC,
            "shared_scene_no_dynamic",
            {},
            {},
            0,
        };
        static const Definition all_shared_mesh_transform = {
            Scenario::ALL_SHARED_MESH_TRANSFORM,
            "all_shared_mesh_transform",
            {
                {"shared-red", {{0.9f, 0.1f, 0.1f}}, 0.34f, 0},
                {"shared-green", {{0.1f, 0.9f, 0.1f}}, 0.34f, 1},
            },
            {
                {0, 0, ALL_CAMERAS, pose(4.0f, -0.9f, 0.0f), pose(3.2f, -0.9f, 0.8f), true, 0, 1},
                {0, 1, ALL_CAMERAS, pose(4.0f, 0.9f, 0.0f), pose(4.0f, 0.9f, 0.0f), false, 1, 2},
            },
            ALL_CAMERAS,
        };
        static const Definition partially_shared_mesh_transform = {
            Scenario::PARTIALLY_SHARED_MESH_TRANSFORM,
            "partially_shared_mesh_transform",
            {
                {"partially-shared", {{0.9f, 0.1f, 0.1f}}, 0.34f, 0},
                {"private-0", {{0.1f, 0.1f, 0.9f}}, 0.34f, 2},
                {"private-1", {{0.1f, 0.1f, 0.9f}}, 0.34f, 2},
                {"private-2", {{0.1f, 0.1f, 0.9f}}, 0.34f, 2},
                {"private-3", {{0.1f, 0.1f, 0.9f}}, 0.34f, 2},
            },
            {
                {0, 0, static_cast<CameraMask>(camera_bit(0) | camera_bit(1)), pose(4.0f, -1.0f, 1.0f), pose(3.2f, -1.0f, 0.4f), true, 0, 1},
                {1, 1, camera_bit(0), pose(4.0f, camera_y(0), -0.9f), pose(4.0f, camera_y(0), -0.9f), false, 0, 5},
                {2, 2, camera_bit(1), pose(4.0f, camera_y(1), -0.9f), pose(4.0f, camera_y(1), -0.9f), false, 0, 9},
                {3, 3, camera_bit(2), pose(4.0f, camera_y(2), -0.9f), pose(4.0f, camera_y(2), -0.9f), false, 0, 13},
                {4, 4, camera_bit(3), pose(4.0f, camera_y(3), -0.9f), pose(4.0f, camera_y(3), -0.9f), false, 0, 17},
            },
            static_cast<CameraMask>(camera_bit(0) | camera_bit(1)),
        };
        static const Definition shared_mesh_individual_transform = {
            Scenario::SHARED_MESH_INDIVIDUAL_TRANSFORM,
            "shared_mesh_individual_transform",
            {
                {"mesh-shared-pose-private", {{0.1f, 0.9f, 0.1f}}, 0.34f, 1},
            },
            {
                {0, 0, camera_bit(0), pose(4.0f, camera_y(0), 0.0f), pose(4.0f, camera_y(0), 0.0f), false, 0, 1},
                {1, 0, camera_bit(1), pose(4.0f, camera_y(1), 0.0f), pose(4.0f, camera_y(1), 0.0f), false, 0, 5},
                {2, 0, camera_bit(2), pose(4.0f, camera_y(2), 0.0f), pose(3.2f, 0.4f, 0.8f), true, 0, 9},
                {3, 0, camera_bit(3), pose(4.0f, camera_y(3), 0.0f), pose(4.0f, camera_y(3), 0.0f), false, 0, 13},
            },
            camera_bit(2),
        };
        static const Definition disjoint = {
            Scenario::DISJOINT,
            "disjoint",
            {
                {"disjoint-0", {{0.9f, 0.1f, 0.1f}}, 0.20f, 0},
                {"disjoint-1", {{0.1f, 0.9f, 0.1f}}, 0.28f, 1},
                {"disjoint-2", {{0.1f, 0.1f, 0.9f}}, 0.36f, 2},
                {"disjoint-3", {{0.9f, 0.1f, 0.1f}}, 0.44f, 0},
            },
            {
                {0, 0, camera_bit(0), pose(4.0f, camera_y(0), 0.0f), pose(4.0f, camera_y(0), 0.0f), false, 0, 1},
                {1, 1, camera_bit(1), pose(4.0f, camera_y(1), 0.0f), pose(3.2f, -0.4f, 0.8f), true, 0, 5},
                {2, 2, camera_bit(2), pose(4.0f, camera_y(2), 0.0f), pose(4.0f, camera_y(2), 0.0f), false, 0, 9},
                {3, 3, camera_bit(3), pose(4.0f, camera_y(3), 0.0f), pose(4.0f, camera_y(3), 0.0f), false, 0, 13},
            },
            camera_bit(1),
        };
        static const Definition mixed = {
            Scenario::MIXED,
            "mixed",
            {
                {"shared-all", {{0.9f, 0.1f, 0.1f}}, 0.34f, 0},
                {"shared-subset", {{0.1f, 0.9f, 0.1f}}, 0.34f, 1},
                {"mesh-only", {{0.1f, 0.1f, 0.9f}}, 0.34f, 2},
                {"mixed-private-0", {{0.9f, 0.1f, 0.1f}}, 0.20f, 0},
                {"mixed-private-1", {{0.9f, 0.1f, 0.1f}}, 0.24f, 0},
                {"mixed-private-2", {{0.9f, 0.1f, 0.1f}}, 0.28f, 0},
                {"mixed-private-3", {{0.9f, 0.1f, 0.1f}}, 0.32f, 0},
            },
            {
                {0, 0, ALL_CAMERAS, pose(4.0f, -1.3f, 1.3f), pose(4.0f, -1.3f, 1.3f), false, 0, 1},
                {1, 1, static_cast<CameraMask>(camera_bit(0) | camera_bit(1)), pose(4.0f, 0.3f, 1.3f), pose(4.0f, 0.3f, 1.3f), false, 0, 5},
                {2, 2, camera_bit(0), pose(4.0f, camera_y(0), 0.0f), pose(4.0f, camera_y(0), 0.0f), false, 0, 9},
                {6, 3, camera_bit(0), pose(4.0f, camera_y(0), -1.3f), pose(4.0f, camera_y(0), -1.3f), false, 0, 25},
                {3, 2, camera_bit(1), pose(4.0f, camera_y(1), 0.0f), pose(4.0f, camera_y(1), 0.0f), false, 0, 13},
                {7, 4, camera_bit(1), pose(4.0f, camera_y(1), -1.3f), pose(4.0f, camera_y(1), -1.3f), false, 0, 29},
                {4, 2, camera_bit(2), pose(4.0f, camera_y(2), 0.0f), pose(3.2f, 0.4f, 0.7f), true, 0, 17},
                {8, 5, camera_bit(2), pose(4.0f, camera_y(2), -1.3f), pose(4.0f, camera_y(2), -1.3f), false, 0, 33},
                {5, 2, camera_bit(3), pose(4.0f, camera_y(3), 0.0f), pose(4.0f, camera_y(3), 0.0f), false, 0, 21},
                {9, 6, camera_bit(3), pose(4.0f, camera_y(3), -1.3f), pose(4.0f, camera_y(3), -1.3f), false, 0, 37},
            },
            camera_bit(2),
        };

        switch(scenario){
            case Scenario::SHARED_SCENE_NO_DYNAMIC: return shared_scene_no_dynamic;
            case Scenario::ALL_SHARED_MESH_TRANSFORM: return all_shared_mesh_transform;
            case Scenario::PARTIALLY_SHARED_MESH_TRANSFORM: return partially_shared_mesh_transform;
            case Scenario::SHARED_MESH_INDIVIDUAL_TRANSFORM: return shared_mesh_individual_transform;
            case Scenario::DISJOINT: return disjoint;
            case Scenario::MIXED: return mixed;
        }
        throw std::invalid_argument("Unknown overlay scenario");
    }

    inline const char* scenario_id(Scenario scenario){
        return definition(scenario).id;
    }

    template <typename T_T, typename T_TI>
    struct StaticConfig: rl_tools::rendering::raytracing::config::Default<T_T, T_TI> {
        static constexpr T_TI CAM_WIDTH = 64;
        static constexpr T_TI CAM_HEIGHT = 64;
        static constexpr T_TI NUM_CAMERAS = static_cast<T_TI>(overlay_scenarios::NUM_CAMERAS);
        static constexpr T_TI NUM_PROBES = 1;
        using SHADING = rl_tools::rendering::raytracing::Low;
        static constexpr bool OUTPUT_RGB = true;
        static constexpr bool OUTPUT_DEPTH = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
    };

    template <typename T_T, typename T_TI>
    using StaticSpecification = rl_tools::rendering::raytracing::Specification<StaticConfig<T_T, T_TI>>;

    template <typename T_T, typename T_TI>
    struct OverlayConfig: StaticConfig<T_T, T_TI> {
        static constexpr T_TI NUM_OVERLAYS = static_cast<T_TI>(overlay_scenarios::NUM_OVERLAYS);
        static constexpr T_TI MAX_OVERLAY_INSTANCES = static_cast<T_TI>(overlay_scenarios::MAX_OVERLAY_INSTANCES);
        static constexpr T_TI MAX_OVERLAYS_PER_CAMERA = static_cast<T_TI>(overlay_scenarios::MAX_OVERLAYS_PER_CAMERA);
    };

    template <typename T_T, typename T_TI>
    using OverlaySpecification = rl_tools::rendering::raytracing::Specification<OverlayConfig<T_T, T_TI>>;

    struct State {
        explicit State(Scenario scenario): scenario(scenario) {}

        Scenario scenario;
        rl_tools::rendering::raytracing::Scene scene;
        rl_tools::rendering::raytracing::AssetPool pool;
        std::vector<rl_tools::rendering::raytracing::AssetHandle> assets;
        std::vector<rl_tools::rendering::raytracing::OverlayPlacement> placements;
        std::vector<std::uint32_t> ids;
        std::array<std::size_t, NUM_CAMERAS> cameras = {{0, 1, 2, 3}};
        bool initial_built = false;
    };

    namespace detail {
        inline rl_tools::rendering::raytracing::Mesh make_quad(float half_extent, const std::array<float, 3>& color, float x = 0.0f, float y = 0.0f, float z = 0.0f){
            rl_tools::rendering::raytracing::Mesh mesh;
            mesh.vertices = {
                x, y - half_extent, z - half_extent,
                x, y + half_extent, z - half_extent,
                x, y + half_extent, z + half_extent,
                x, y - half_extent, z + half_extent,
            };
            mesh.indices = {0, 2, 1, 0, 3, 2};
            for(std::size_t channel = 0; channel < 3; channel++){
                mesh.color[channel] = color[channel];
            }
            return mesh;
        }

        template <typename DEVICE>
        void add_shared_scene(DEVICE& device, State& state){
            rl_tools::rendering::raytracing::Object background;
            background.name = "shared-scene";
            background.meshes.push_back(make_quad(10.0f, {{0.2f, 0.2f, 0.2f}}));
            background.meshes.push_back(make_quad(0.55f, {{0.75f, 0.15f, 0.15f}}, -0.8f, -2.0f, 1.5f));
            background.meshes.push_back(make_quad(0.42f, {{0.15f, 0.65f, 0.2f}}, -0.5f, 1.7f, -1.1f));
            background.meshes.push_back(make_quad(0.30f, {{0.15f, 0.25f, 0.8f}}, -1.0f, 0.4f, 2.3f));
            const auto transform = pose(8.0f, 0.0f, 0.0f);
            rl_tools::add(device, state.scene, background, transform.data());
        }

        template <typename DEVICE>
        rl_tools::rendering::raytracing::AssetHandle add_asset(DEVICE& device, State& state, const AssetDefinition& asset){
            rl_tools::rendering::raytracing::Object object;
            object.name = asset.name;
            object.meshes.push_back(make_quad(asset.half_extent, asset.color));
            return rl_tools::add(device, state.pool, object);
        }
    }

    template <typename DEVICE>
    State prepare(DEVICE& device, Scenario scenario){
        State state(scenario);
        detail::add_shared_scene(device, state);
        const auto& scenario_definition = definition(scenario);
        state.assets.reserve(scenario_definition.assets.size());
        for(const auto& asset : scenario_definition.assets){
            state.assets.push_back(detail::add_asset(device, state, asset));
        }
        return state;
    }

    template <typename TI>
    constexpr std::array<TI, NUM_CAMERAS> identity_camera_mapping(){
        return {{static_cast<TI>(0), static_cast<TI>(1), static_cast<TI>(2), static_cast<TI>(3)}};
    }

    template <typename DEVICE, typename SPEC>
    void build_initial(DEVICE& device, rl_tools::rendering::raytracing::Renderer<SPEC>& renderer, State& state, const std::array<typename SPEC::TI, NUM_CAMERAS>& camera_mapping){
        static_assert(SPEC::ENABLE_OVERLAYS, "Overlay scenarios require an overlay specification");
        static_assert(static_cast<std::size_t>(SPEC::NUM_CAMERAS) >= NUM_CAMERAS, "Overlay scenarios require four mapped cameras");
        static_assert(static_cast<std::size_t>(SPEC::NUM_OVERLAYS) >= NUM_OVERLAYS, "Overlay scenarios require ten overlays");
        static_assert(static_cast<std::size_t>(SPEC::MAX_OVERLAY_INSTANCES) == MAX_OVERLAY_INSTANCES, "Overlay scenario IDs require four slots per overlay");
        static_assert(static_cast<std::size_t>(SPEC::MAX_OVERLAYS_PER_CAMERA) >= MAX_OVERLAYS_PER_CAMERA, "Overlay scenarios require four overlays per camera");
        if(state.initial_built){
            throw std::logic_error("Overlay scenario initial state was already built");
        }
        const auto& scenario_definition = definition(state.scenario);
        std::array<bool, SPEC::NUM_OVERLAYS> attached = {};
        state.placements.reserve(scenario_definition.placements.size());
        state.ids.reserve(scenario_definition.placements.size());
        for(std::size_t logical_camera = 0; logical_camera < NUM_CAMERAS; logical_camera++){
            state.cameras[logical_camera] = static_cast<std::size_t>(camera_mapping[logical_camera]);
        }
        for(const auto& placement_definition : scenario_definition.placements){
            if(!attached[placement_definition.overlay]){
                for(std::size_t logical_camera = 0; logical_camera < NUM_CAMERAS; logical_camera++){
                    if((placement_definition.cameras & camera_bit(logical_camera)) != 0){
                        rl_tools::attach(device, renderer, camera_mapping[logical_camera], rl_tools::rendering::raytracing::OverlayIndex{placement_definition.overlay});
                    }
                }
                attached[placement_definition.overlay] = true;
            }
            const auto placement = rl_tools::spawn(device, renderer, rl_tools::rendering::raytracing::OverlayIndex{placement_definition.overlay}, state.assets[placement_definition.asset], placement_definition.initial_transform.data());
            if(placement.first_slot != placement_definition.expected_first_slot){
                throw std::logic_error("Overlay scenario placement slot changed");
            }
            const auto id = static_cast<std::uint32_t>(state.scene.instances.size() + placement_definition.overlay * SPEC::MAX_OVERLAY_INSTANCES + placement.first_slot);
            if(id != placement_definition.expected_id){
                throw std::logic_error("Overlay scenario instance ID changed");
            }
            state.placements.push_back(placement);
            state.ids.push_back(id);
        }
        state.initial_built = true;
    }

    template <typename DEVICE, typename SPEC>
    void build_initial(DEVICE& device, rl_tools::rendering::raytracing::Renderer<SPEC>& renderer, State& state){
        build_initial(device, renderer, state, identity_camera_mapping<typename SPEC::TI>());
    }

    template <typename DEVICE, typename SPEC>
    void apply_update(DEVICE& device, rl_tools::rendering::raytracing::Renderer<SPEC>& renderer, const State& state){
        if(!state.initial_built){
            throw std::logic_error("Overlay scenario initial state has not been built");
        }
        const auto& scenario_definition = definition(state.scenario);
        for(std::size_t placement_i = 0; placement_i < scenario_definition.placements.size(); placement_i++){
            const auto& placement_definition = scenario_definition.placements[placement_i];
            if(placement_definition.moves_in_updated_state){
                rl_tools::set_transform(device, renderer, rl_tools::rendering::raytracing::OverlayIndex{placement_definition.overlay}, state.placements[placement_i], placement_definition.updated_transform.data());
            }
        }
    }
}

#endif
