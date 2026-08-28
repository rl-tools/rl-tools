#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>

#include "../../../utils/utils.h"

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>

namespace rlt = rl_tools;
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
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 2;
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        using SHADING = rlt::rendering::raytracing::Low;
    };
    using WORLD = rlt::rl::environments::hyperdrone::World<WORLD_SPEC>;
    using ENVIRONMENT = rlt::rl::environments::hyperdrone::MultiEnvironment<WORLD, 1>;
    using ANNOTATIONS = typename WORLD::ANNOTATIONS;
    using GLB = rlt::rendering::datasets::procthor::GLB;
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
    auto env = std::make_unique<ENVIRONMENT>();
    rlt::malloc(device, *env);
    env->shared.annotation_cache.directory = cache_directory;
    rlt::init(device, *env, dataset);
    annotations_out = env->environments[0].slots[0].annotations;
    rlt::free(device, *env);
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

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, PARAMETER_KEYING) {
    DEVICE device;
    rlt::init(device);
    const GLB dataset{scene_directory(), {}};
    const std::string cache_directory = fresh_cache_directory("rl_tools_annotations_cache_keying");
    const rlt::rendering::datasets::annotations::Cache cache{cache_directory};

    auto env = std::make_unique<ENVIRONMENT>();
    rlt::malloc(device, *env);
    rlt::init(device, *env, dataset);
    auto& slot = env->environments[0].slots[0];

    // different parameters are different cache entries; each warm read is bit-identical to the
    // uncached scan with the same parameters
    rlt::rendering::datasets::annotations::FreeSpaceParameters<T, TI> parameters{};
    for (const TI min_required : {(TI)20, (TI)100}) {
        parameters.min_required_positions = min_required;
        ANNOTATIONS cold, warm, uncached;
        rlt::rendering::datasets::annotations::annotate(device, cold, slot.metadata, slot.renderer, parameters, cache);
        rlt::rendering::datasets::annotations::annotate(device, warm, slot.metadata, slot.renderer, parameters, cache);
        rlt::rendering::datasets::annotations::annotate(device, uncached, slot.metadata, slot.renderer, parameters);
        EXPECT_EQ(cold.num_positions, (TI)min_required);
        EXPECT_TRUE(annotations_equal(cold, warm));
        EXPECT_TRUE(annotations_equal(cold, uncached));
    }
    EXPECT_EQ(cache_entry_count(cache_directory), (size_t)2);
    rlt::free(device, *env);
}

namespace test_probe_batch_independence {
    template <TI T_NUM_CAMERAS>
    struct PROBE_CONFIG: rlt::rendering::raytracing::config::Default<T, TI> {
        static constexpr TI CAM_WIDTH = 32;
        static constexpr TI CAM_HEIGHT = 32;
        static constexpr TI NUM_CAMERAS = T_NUM_CAMERAS;
        static constexpr TI NUM_PROBES = 8;
        using SHADING = rlt::rendering::raytracing::Low;
        static constexpr bool OUTPUT_RGB = true;
        static constexpr bool OUTPUT_DEPTH = false;
        static constexpr bool OUTPUT_SEGMENTATION = false;
    };

    template <TI NUM_CAMERAS, typename SPEC>
    void annotate_standalone(DEVICE& device, rlt::rendering::datasets::annotations::FreeSpace<SPEC>& annotations){
        using RENDERER = rlt::rendering::raytracing::Renderer<rlt::rendering::raytracing::Specification<PROBE_CONFIG<NUM_CAMERAS>>>;
        auto renderer = std::make_unique<RENDERER>();
        rlt::rendering::Bundle<T> bundle;
        ASSERT_TRUE((rlt::load<rlt::rendering::raytracing::Low, true>(device, bundle, SCENE_PATH)));
        rlt::malloc(device, *renderer);
        rlt::init(device, *renderer, bundle);
        rlt::generate_probe_directions(device, *renderer);
        rlt::rendering::datasets::annotations::FreeSpaceParameters<T, TI> parameters{};
        rlt::rendering::datasets::annotations::annotate(device, annotations, bundle.metadata, *renderer, parameters);
        rlt::free(device, *renderer);
    }
}

TEST(RL_ENVIRONMENTS_HYPERDRONE_ANNOTATIONS, PROBE_BATCH_INDEPENDENCE) {
    using namespace test_probe_batch_independence;
    DEVICE device;
    rlt::init(device);
    // the scan is candidate-exact: the probe vehicle's batch width must not affect the table
    ANNOTATIONS narrow, wide;
    annotate_standalone<2>(device, narrow);
    annotate_standalone<7>(device, wide);
    EXPECT_GT(narrow.num_positions, (TI)0);
    EXPECT_TRUE(annotations_equal(narrow, wide));
}
