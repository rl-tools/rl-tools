#include <rl_tools/operations/cpu_mux.h>
#include "../../../../../src/rl/environments/hyperdrone/tasks/target_frame/yaw/scenes.h"
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <string>

namespace rlt = rl_tools;
namespace rro = rlt::hyperlux::outputs;
namespace l2f = rlt::rl::environments::l2f;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using T = float;
using TI = typename DEVICE::index_t;

#ifdef RL_TOOLS_TEST_DATA_PATH
static const std::string SCENE_PATH = std::string(RL_TOOLS_MACRO_TO_STR(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
static const std::string SCENE_PATH = "";
#endif

namespace test_hyperdrone_annotations {
    using REWARD_FUNCTION = l2f::parameters::reward_functions::Squared<T>;
    static constexpr TI EPISODE_STEP_LIMIT = 500;
    using PARAMETERS_SPEC = l2f::ParametersBaseSpecification<T, TI, 4, EPISODE_STEP_LIMIT, REWARD_FUNCTION>;
    using PARAMETERS_TYPE = l2f::ParametersDisturbances<l2f::ParametersSpecification<T, TI, l2f::ParametersBase<PARAMETERS_SPEC>>>;

    struct DYNAMICS_STATIC_PARAMETERS {
        static constexpr auto ACTION_INTERFACE = l2f::parameters::ActionInterface::DIRECT_MOTOR;
        static constexpr TI N_SUBSTEPS = 1;
        static constexpr TI ACTION_HISTORY_LENGTH = 1;
        static constexpr TI CLOSED_FORM = false;
        using STATE_BASE = l2f::StateBase<l2f::StateSpecification<T, TI>>;
        using STATE_TYPE = l2f::StateRotorsHistory<l2f::StateRotorsHistorySpecification<T, TI, ACTION_HISTORY_LENGTH, CLOSED_FORM, l2f::StateRandomForce<l2f::StateSpecification<T, TI, l2f::StateLastAction<l2f::StateSpecification<T, TI, STATE_BASE>>>>>>;
        using OBSERVATION_TYPE = l2f::observation::Position<l2f::observation::PositionSpecification<T, TI,
                l2f::observation::OrientationRotationMatrix<l2f::observation::OrientationRotationMatrixSpecification<T, TI,
                l2f::observation::LinearVelocity<l2f::observation::LinearVelocitySpecification<T, TI,
                l2f::observation::AngularVelocity<l2f::observation::AngularVelocitySpecification<T, TI>>>>>>>>;
        using OBSERVATION_TYPE_PRIVILEGED = OBSERVATION_TYPE;
        static constexpr bool PRIVILEGED_OBSERVATION_NOISE = false;
        using PARAMETERS = PARAMETERS_TYPE;
        static constexpr auto dynamics = l2f::parameters::dynamics::registry<l2f::parameters::dynamics::REGISTRY::crazyflie, PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::Integration integration = {(T)0.01};
        static constexpr typename PARAMETERS_TYPE::MDP::Initialization init = l2f::parameters::init::init_90_deg<PARAMETERS_SPEC>;
        static constexpr typename PARAMETERS_TYPE::MDP mdp = {init, REWARD_FUNCTION{}, {}, {}, {}};
        static constexpr typename PARAMETERS_TYPE::Disturbances disturbances = {{0, 0}, {0, 0}};
        static constexpr PARAMETERS_TYPE PARAMETER_VALUES = {{dynamics, integration, mdp}, disturbances};
        static constexpr T STATE_LIMIT_POSITION_X = 100000;
        static constexpr T STATE_LIMIT_POSITION_Y = 100000;
        static constexpr T STATE_LIMIT_POSITION_Z = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_VELOCITY_Z = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_X = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Y = 100000;
        static constexpr T STATE_LIMIT_ANGULAR_VELOCITY_Z = 100000;
    };

    struct WORLD_SPEC: rlt::rl::environments::hyperdrone::Specification<T, TI, DYNAMICS_STATIC_PARAMETERS> {
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI INSTANCES_PER_WORLD = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::hyperlux::Low;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    using ENVIRONMENT = rlt::rl::environments::HyperDrone<WORLD, 1>;
    using ANNOTATIONS = typename WORLD::ANNOTATIONS;
    using GLB = rlt::hyperlux::datasets::procthor::GLB;
}

using namespace test_hyperdrone_annotations;

static std::string scene_directory(){
    std::string directory = std::filesystem::temp_directory_path() / "rl_tools_hyperdrone_annotations_scenes";
    std::filesystem::create_directories(directory);
    std::filesystem::path link = std::filesystem::path(directory) / "scene_0.glb";
    if (!std::filesystem::exists(link)) {
        std::filesystem::create_symlink(SCENE_PATH, link);
    }
    return directory;
}

static std::string fresh_cache_directory(const std::string& name){
    std::filesystem::path directory = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(directory);
    return directory.string();
}

static bool annotations_equal(const ANNOTATIONS& a, const ANNOTATIONS& b){
    return a.num_positions == b.num_positions
        && std::memcmp(a.positions, b.positions, sizeof(a.positions[0]) * a.num_positions) == 0;
}

template <typename DATASET>
static void init_and_capture(DEVICE& device, const DATASET& dataset, const std::string& cache_directory, ANNOTATIONS& annotations_out){
    ENVIRONMENT env;
    rlt::malloc(device, env);
    env.shared.annotation_cache.directory = cache_directory;
    rlt::init(device, env, dataset);
    annotations_out = env.worlds[0].slots[0].annotations;
    rlt::free(device, env);
}

static size_t cache_entry_count(const std::string& cache_directory){
    const std::filesystem::path directory = std::filesystem::path(cache_directory) / "free_space";
    if (!std::filesystem::is_directory(directory)) {
        return 0;
    }
    size_t count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        if (entry.is_regular_file() && entry.path().extension() == ".bin") {
            count++;
        }
    }
    return count;
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, CACHE_IDENTITY) {
    DEVICE device;
    rlt::init(device);
    const GLB dataset{scene_directory(), {}};
    const std::string cache_directory = fresh_cache_directory("rl_tools_annotations_cache_identity");
    ANNOTATIONS uncached, cold, warm;
    init_and_capture(device, dataset, "", uncached);
    init_and_capture(device, dataset, cache_directory, cold);
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)1);
    init_and_capture(device, dataset, cache_directory, warm);
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)1);
    EXPECT_GT(uncached.num_positions, (TI)0);
    EXPECT_TRUE(annotations_equal(uncached, cold));
    EXPECT_TRUE(annotations_equal(cold, warm));
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, YAW_SCENE_LIGHTING_ADAPTER) {
    using namespace rl_tools;
    namespace yaw = rl::environments::hyperdrone::tasks::target_frame::yaw;
    DEVICE device;
    init(device);
    yaw::Scenes dataset;
    dataset.references.push_back(SCENE_PATH);
    yaw::Scenes::Corpus corpus;
    enumerate(device, dataset, corpus);
    hyperlux::Bundle<T> normal, adapted;
    ASSERT_TRUE((load<hyperlux::High, true>(device, static_cast<const GLB&>(dataset), corpus, 0, normal)));
    ASSERT_EQ(normal.scene.lights.size(), (size_t)5);
    size_t normal_unlit = 0;
    for(const auto& object: normal.scene.objects){
        for(const auto& mesh: object.meshes){
            normal_unlit += mesh.unlit;
        }
    }
    ASSERT_GT(normal_unlit, (size_t)0);
    ASSERT_TRUE((load<hyperlux::High, true>(device, dataset, corpus, 0, adapted)));
    ASSERT_EQ(adapted.scene.lights.size(), normal.scene.lights.size() + 3);
    const float directions[3][3] = {{-0.70710678f, 0, 0.70710678f}, {0, -0.70710678f, 0.70710678f}, {0, 0.70710678f, 0.70710678f}};
    const float intensities[3] = {0.4f, 0.3f, 0.2f};
    for(size_t light = 0; light < 3; light++){
        EXPECT_EQ(adapted.scene.lights[light].type, 0);
        for(size_t axis = 0; axis < 3; axis++){
            EXPECT_FLOAT_EQ(adapted.scene.lights[light].direction[axis], directions[light][axis]);
            EXPECT_FLOAT_EQ(adapted.scene.lights[light].color[axis], intensities[light]);
        }
    }
    for(size_t light = 0; light < normal.scene.lights.size(); light++){
        EXPECT_EQ(std::memcmp(&normal.scene.lights[light], &adapted.scene.lights[light + 3], sizeof(hyperlux::SceneLight)), 0);
    }
    EXPECT_EQ(normal.scene.environment.mode, hyperlux::Scene::Environment::Mode::DEFAULT);
    EXPECT_EQ(adapted.scene.environment.mode, hyperlux::Scene::Environment::Mode::SOLID);
    for(size_t axis = 0; axis < 3; axis++){
        EXPECT_FLOAT_EQ(adapted.scene.environment.ambient[axis], 0.5f);
        EXPECT_FLOAT_EQ(adapted.scene.environment.horizon[axis], 0.0f);
        EXPECT_EQ(adapted.metadata.center[axis], normal.metadata.center[axis]);
        EXPECT_EQ(adapted.metadata.half_extent[axis], normal.metadata.half_extent[axis]);
    }
    EXPECT_EQ(adapted.metadata.max_ray_length, normal.metadata.max_ray_length);
    EXPECT_EQ(adapted.metadata.content_hash, normal.metadata.content_hash);
    for(const auto& object: adapted.scene.objects){
        for(const auto& mesh: object.meshes){
            EXPECT_FALSE(mesh.unlit);
        }
    }
    size_t normal_unlit_after = 0;
    for(const auto& object: normal.scene.objects){
        for(const auto& mesh: object.meshes){
            normal_unlit_after += mesh.unlit;
        }
    }
    EXPECT_EQ(normal_unlit_after, normal_unlit);
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, PARAMETER_KEYING) {
    DEVICE device;
    rlt::init(device);
    const GLB dataset{scene_directory(), {}};
    const std::string cache_directory = fresh_cache_directory("rl_tools_annotations_cache_keying");
    const rlt::hyperlux::datasets::annotations::Cache cache{cache_directory};

    ENVIRONMENT env;
    rlt::malloc(device, env);
    rlt::init(device, env, dataset);
    auto& slot = env.worlds[0].slots[0];

    // different parameters are different cache entries; each warm read is bit-identical to the
    // uncached scan with the same parameters
    rlt::hyperlux::datasets::annotations::FreeSpaceParameters<T, TI> parameters{};
    for (const TI min_required : {(TI)20, (TI)100}) {
        parameters.min_required_positions = min_required;
        ANNOTATIONS cold, warm, uncached;
        rlt::hyperlux::datasets::annotations::annotate(device, cold, slot.metadata, slot.renderer, parameters, cache);
        rlt::hyperlux::datasets::annotations::annotate(device, warm, slot.metadata, slot.renderer, parameters, cache);
        rlt::hyperlux::datasets::annotations::annotate(device, uncached, slot.metadata, slot.renderer, parameters);
        EXPECT_EQ(cold.num_positions, (TI)min_required);
        EXPECT_TRUE(annotations_equal(cold, warm));
        EXPECT_TRUE(annotations_equal(cold, uncached));
    }
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)2);
    rlt::free(device, env);
}

namespace test_probe_batch_independence {
    template <TI T_NUM_CAMERAS, bool SPARSE = false>
    struct PROBE_CONFIG: rlt::hyperlux::config::Default<T, TI> {
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
        static constexpr TI NUM_PROBES = 8;
        using SHADING = rlt::hyperlux::Low;
        using OUTPUTS = rro::List<rlt::utils::typing::conditional_t<SPARSE, rro::Cameras<rro::cameras::Every<3, 1>, rro::Probes>, rro::Probes>>;
    };

    template <TI NUM_CAMERAS, bool SPARSE = false, typename SPEC>
    void annotate_standalone(DEVICE& device, rlt::hyperlux::datasets::annotations::FreeSpace<SPEC>& annotations){
        using RENDERER = rlt::HyperLux<rlt::hyperlux::Specification<PROBE_CONFIG<NUM_CAMERAS, SPARSE>>>;
        RENDERER renderer;
        rlt::hyperlux::Bundle<T> bundle;
        ASSERT_TRUE((rlt::load<rlt::hyperlux::Low, true>(device, bundle, SCENE_PATH)));
        rlt::malloc(device, renderer);
        rlt::init(device, renderer, bundle);
        rlt::generate_probe_directions(device, renderer);
        rlt::hyperlux::datasets::annotations::FreeSpaceParameters<T, TI> parameters{};
        rlt::hyperlux::datasets::annotations::annotate(device, annotations, bundle.metadata, renderer, parameters);
        rlt::free(device, renderer);
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, PROBE_BATCH_INDEPENDENCE) {
    using namespace test_probe_batch_independence;
    DEVICE device;
    rlt::init(device);
    // the scan is candidate-exact: the probe vehicle's batch width must not affect the table
    ANNOTATIONS single, narrow, wide, sparse;
    annotate_standalone<1>(device, single);
    annotate_standalone<2>(device, narrow);
    annotate_standalone<7>(device, wide);
    annotate_standalone<7, true>(device, sparse);
    EXPECT_GT(narrow.num_positions, (TI)0);
    EXPECT_TRUE(annotations_equal(single, narrow));
    EXPECT_TRUE(annotations_equal(narrow, wide));
    EXPECT_TRUE(annotations_equal(narrow, sparse));
}

namespace test_indoor_regions {
    rlt::hyperlux::SceneRegion region(std::initializer_list<std::array<float, 2>> polygon, float min_height, float max_height, bool indoor = true){
        rlt::hyperlux::SceneRegion result;
        result.polygon = polygon;
        result.indoor = indoor;
        result.min[0] = result.min[1] = std::numeric_limits<float>::max();
        result.max[0] = result.max[1] = std::numeric_limits<float>::lowest();
        result.min[2] = min_height;
        result.max[2] = max_height;
        for(const auto& vertex: polygon){
            for(size_t axis = 0; axis < 2; axis++){
                result.min[axis] = std::min(result.min[axis], vertex[axis]);
                result.max[axis] = std::max(result.max[axis], vertex[axis]);
            }
        }
        return result;
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, REGION_CONCAVITY_BOUNDARIES_AND_HEIGHT) {
    using namespace rl_tools;
    DEVICE device;
    auto volume = test_indoor_regions::region({{0, 0}, {4, 0}, {4, 1}, {1, 1}, {1, 4}, {0, 4}}, 1, 3);
    const T inside[][3] = {{0.5, 3.5, 2}, {3.5, 0.5, 2}, {0, 2, 2}, {2, 0, 2}, {1, 1, 2}, {0.5, 0.5, 1}, {0.5, 0.5, 3}};
    const T outside[][3] = {{2, 2, 2}, {-0.01, 0.5, 2}, {0.5, 4.01, 2}, {0.5, 0.5, 0.99}, {0.5, 0.5, 3.01}};
    for(size_t winding = 0; winding < 2; winding++){
        for(const auto& point: inside){
            EXPECT_TRUE(contains(device, volume, point));
        }
        for(const auto& point: outside){
            EXPECT_FALSE(contains(device, volume, point));
        }
        std::reverse(volume.polygon.begin(), volume.polygon.end());
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, INDOOR_REGION_REQUIREMENT_AND_OUTDOOR_OVERLAP) {
    using namespace rl_tools;
    DEVICE device;
    hyperlux::SceneRegions regions;
    const T overlap[3] = {0.5, 0.5, 2};
    const T indoor_only[3] = {3.5, 0.5, 2};
    const T above_outdoor[3] = {0.5, 0.5, 2.75};
    const T unannotated[3] = {10, 10, 2};
    EXPECT_TRUE(contains_indoor_position(device, regions, unannotated));
    regions.annotated = true;
    EXPECT_FALSE(contains_indoor_position(device, regions, overlap));
    regions.volumes.push_back(test_indoor_regions::region({{0, 0}, {1, 0}, {1, 1}, {0, 1}}, 1.5, 2.5, false));
    EXPECT_FALSE(contains_indoor_position(device, regions, overlap));
    EXPECT_FALSE(contains_indoor_position(device, regions, indoor_only));
    regions.volumes.push_back(test_indoor_regions::region({{0, 0}, {4, 0}, {4, 4}, {0, 4}}, 1, 3));
    for(size_t order = 0; order < 2; order++){
        EXPECT_FALSE(contains_indoor_position(device, regions, overlap));
        EXPECT_FALSE(contains_indoor_position(device, regions, unannotated));
        EXPECT_TRUE(contains_indoor_position(device, regions, indoor_only));
        EXPECT_TRUE(contains_indoor_position(device, regions, above_outdoor));
        std::reverse(regions.volumes.begin(), regions.volumes.end());
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, REGION_CACHE_KEYING) {
    namespace annotations = rlt::hyperlux::datasets::annotations;
    rlt::hyperlux::SceneMetadata<T> metadata;
    annotations::FreeSpaceParameters<T, TI> parameters;
    const auto key = [&](const rlt::hyperlux::SceneMetadata<T>& value){
        return annotations::free_space::key<T>(value, 8, parameters);
    };
    const std::string unrestricted_key = key(metadata);
    metadata.regions.annotated = true;
    EXPECT_NE(unrestricted_key, key(metadata));
    const std::string empty_key = key(metadata);
    metadata.regions.volumes.push_back(test_indoor_regions::region({{0, 0}, {4, 0}, {4, 4}, {0, 4}}, 1, 3));
    const std::string region_key = key(metadata);
    EXPECT_NE(empty_key, region_key);
    auto changed = metadata;
    changed.regions.volumes[0].indoor = false;
    EXPECT_NE(region_key, key(changed));
    changed = metadata;
    changed.regions.volumes[0].polygon[1][0] = 3;
    EXPECT_NE(region_key, key(changed));
    for(size_t axis = 0; axis < 3; axis++){
        changed = metadata;
        changed.regions.volumes[0].min[axis] += 0.25;
        EXPECT_NE(region_key, key(changed));
        changed = metadata;
        changed.regions.volumes[0].max[axis] -= 0.25;
        EXPECT_NE(region_key, key(changed));
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, INDOOR_SCAN_AND_CACHE) {
    using namespace rl_tools;
    namespace annotations = hyperlux::datasets::annotations;
    using RENDERER = HyperLux<hyperlux::Specification<test_probe_batch_independence::PROBE_CONFIG<7>>>;
    DEVICE device;
    init(device);
    hyperlux::Bundle<T> bundle;
    ASSERT_TRUE((load<hyperlux::Low, true>(device, bundle, SCENE_PATH)));
    RENDERER renderer;
    malloc(device, renderer);
    init(device, renderer, bundle);
    generate_probe_directions(device, renderer);
    const std::string cache_directory = fresh_cache_directory("rl_tools_annotations_indoor_cache");
    const annotations::Cache cache{cache_directory};
    annotations::FreeSpaceParameters<T, TI> parameters;
    std::vector<annotations::FreeSpacePosition<T>> baseline, filtered, warm;
    annotations::annotate(device, baseline, bundle.metadata, renderer, parameters, cache);
    ASSERT_EQ(baseline.size(), parameters.min_required_positions);
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)1);

    T minimum[3], maximum[3];
    for(size_t axis = 0; axis < 3; axis++){
        minimum[axis] = bundle.metadata.center[axis] - bundle.metadata.half_extent[axis];
        maximum[axis] = bundle.metadata.center[axis] + bundle.metadata.half_extent[axis];
    }
    const T* excluded = baseline[0].position;
    constexpr T margin = 0.05;
    bundle.metadata.regions.annotated = true;
    bundle.metadata.regions.volumes = {
        test_indoor_regions::region({{minimum[0], minimum[1]}, {maximum[0], minimum[1]}, {maximum[0], maximum[1]}, {minimum[0], maximum[1]}}, minimum[2], maximum[2]),
        test_indoor_regions::region({{excluded[0] - margin, excluded[1] - margin}, {excluded[0] + margin, excluded[1] - margin}, {excluded[0] + margin, excluded[1] + margin}, {excluded[0] - margin, excluded[1] + margin}}, excluded[2] - margin, excluded[2] + margin, false)
    };
    annotations::annotate(device, filtered, bundle.metadata, renderer, parameters, cache);
    annotations::annotate(device, warm, bundle.metadata, renderer, parameters, cache);
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)2);
    EXPECT_EQ(filtered.size(), parameters.min_required_positions);
    ASSERT_EQ(filtered.size(), warm.size());
    for(size_t position_i = 0; position_i < filtered.size(); position_i++){
        const auto& position = filtered[position_i];
        bool excluded_volume = true;
        for(size_t axis = 0; axis < 3; axis++){
            EXPECT_GE(position.position[axis], minimum[axis]);
            EXPECT_LE(position.position[axis], maximum[axis]);
            excluded_volume = excluded_volume && position.position[axis] >= excluded[axis] - margin && position.position[axis] <= excluded[axis] + margin;
        }
        EXPECT_FALSE(excluded_volume);
        EXPECT_EQ(std::memcmp(&position, &warm[position_i], sizeof(position)), 0);
    }
    free(device, renderer);
}
