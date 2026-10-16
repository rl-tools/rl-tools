#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/controllers/raptor/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/catalog.h>

#include <emscripten.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace rl_tools::rl::environments::hyperdrone::web {
    using DEVICE = devices::DEVICE_FACTORY<>;
    using TI = DEVICE::index_t;
    using T = float;
    using TYPE_POLICY = numeric_types::Policy<T>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    template <typename T_SHADING, TI T_MOTION_BLUR_SAMPLES, TI T_ANTI_ALIASING_GRID_SIZE, TI T_RESOLUTION_SCALE, TI T_N_AGENTS>
    struct Config: presets::X500FPV<T, TI> {
        using SHADING = T_SHADING;
        static_assert(utils::typing::is_same_v<SHADING, rendering::raytracing::Low>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::Medium>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::High>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::VeryHigh>);
        static_assert(T_MOTION_BLUR_SAMPLES == 1 || T_MOTION_BLUR_SAMPLES == 4);
        static_assert(T_ANTI_ALIASING_GRID_SIZE == 1 || T_ANTI_ALIASING_GRID_SIZE == 2);
        static_assert(T_RESOLUTION_SCALE == 1 || T_RESOLUTION_SCALE == 2);
        static_assert(T_N_AGENTS == 1 || T_N_AGENTS == 8);
        static constexpr const char* FIDELITY = utils::typing::is_same_v<SHADING, rendering::raytracing::Low> ? "low"
            : utils::typing::is_same_v<SHADING, rendering::raytracing::Medium> ? "medium"
            : utils::typing::is_same_v<SHADING, rendering::raytracing::High> ? "high" : "very-high";
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
        static constexpr TI N_AGENTS = T_N_AGENTS;
        static constexpr TI N_CAMERA_AGENTS = 1;
        static constexpr TI MAX_DRONES = 6;
        static constexpr TI MAX_ENTITY_SLOTS_PER_INSTANCE = N_AGENTS * rig::Rotorcraft<TI>::SPEC::MAX_PARTS;
        static constexpr TI CAM_WIDTH = 480 * T_RESOLUTION_SCALE;
        static constexpr TI CAM_HEIGHT = 270 * T_RESOLUTION_SCALE;
        static constexpr TI NUM_PROBES = 64;
        using CAMERA_OUTPUTS = CameraOutputs<
            rendering::raytracing::outputs::List<
                rendering::raytracing::outputs::RGB<>,
                rendering::raytracing::outputs::Depth<rendering::raytracing::outputs::format::RGBA8>,
                rendering::raytracing::outputs::Segmentation<rendering::raytracing::outputs::format::RGBA8, rendering::raytracing::outputs::labels::Semantic>,
                rendering::raytracing::outputs::Segmentation<rendering::raytracing::outputs::format::UInt32, rendering::raytracing::outputs::labels::Semantic>,
                rendering::raytracing::outputs::Normals<rendering::raytracing::outputs::format::RGBA8>,
                rendering::raytracing::outputs::Flow<rendering::raytracing::outputs::direction::Backward, rendering::raytracing::outputs::format::RGBA8>,
                rendering::raytracing::outputs::Probes>,
            rendering::raytracing::outputs::List<
                rendering::raytracing::outputs::RGB<>,
                rendering::raytracing::outputs::Probes>>;
        static constexpr bool ENABLE_MOTION_BLUR = T_MOTION_BLUR_SAMPLES > 1;
        static constexpr TI MOTION_BLUR_SAMPLES = T_MOTION_BLUR_SAMPLES;
        static constexpr bool ENABLE_ANTI_ALIASING = T_ANTI_ALIASING_GRID_SIZE > 1;
        static constexpr TI ANTI_ALIASING_GRID_SIZE = T_ANTI_ALIASING_GRID_SIZE;
        static constexpr T SHUTTER_FRACTION_MIN = 0.5;
        static constexpr T SHUTTER_FRACTION_MAX = 0.5;
    };
    template <typename... CONFIGS>
    struct Configurations {
        static_assert(sizeof...(CONFIGS) > 0, "The web demo requires at least one compiled configuration");
    };
    constexpr TI VIEWS = 6;
    constexpr const char* CANVASES[VIEWS] = {"#camera0", "#camera1", "#depth", "#segmentation", "#normals", "#flow"};
    constexpr TI SPAWN_ATTEMPTS = 4096;
    constexpr T DRONE_SEPARATION = 1.5;
    constexpr double FRAME_PERIOD = 1.0 / 60.0;

    struct Drone {
        std::string path;
        int dynamics_index;
        T fpv_position[3];
    };

    template <typename CONFIG>
    struct State {
        using WORLD = World<CONFIG>;
        using ENVIRONMENT = MultiEnvironment<WORLD, 1>;
        using PILOT = controllers::raptor::Config::Bind<TYPE_POLICY, TI, CONFIG::N_AGENTS>;
        ENVIRONMENT environment;
        PILOT pilot;
        typename PILOT::State pilot_state;
        typename PILOT::Buffer pilot_buffer;
        RNG rng;
        Tensor<tensor::Specification<typename WORLD::Parameters, TI, tensor::Shape<TI, 1>>> parameters;
        Tensor<tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, 1>>> states, next_states;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, WORLD::ACTION_DIM>>> actions;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 1>>> reset_mask;
        controllers::raptor::Target<T> targets[CONFIG::N_AGENTS];
        rendering::datasets::annotations::FreeSpacePosition<T> spawn;
        std::vector<rendering::datasets::annotations::FreeSpacePosition<T>> other_spawns;
        uint32_t seed = 1337;
        T position_offset = 0.3;
        T velocity_target = 0.7;
        T yaw_rate_limit = PILOT::SPEC::YAW_RATE_LIMIT;
        T yaw_error_limit = PILOT::SPEC::YAW_ERROR_LIMIT;
        bool moving = false;
        bool moving_vertical = false;
    };
    template <typename CONFIG>
    struct Buffer {
        static constexpr TI PIXELS = CONFIG::CAM_WIDTH * CONFIG::CAM_HEIGHT;
        std::vector<uint32_t> segmentation = std::vector<uint32_t>(PIXELS);
    };
}

#include <hyperdrone_web_configurations.h>

namespace web = rl_tools::rl::environments::hyperdrone::web;

namespace rl_tools {
    template <typename CONFIG>
    void malloc(web::DEVICE& device, web::State<CONFIG>& state) {
        malloc(device, state.environment);
        malloc(device, state.pilot);
        malloc(device, state.pilot_state);
        malloc(device, state.pilot_buffer);
        malloc(device, state.rng);
        malloc(device, state.parameters);
        malloc(device, state.states);
        malloc(device, state.next_states);
        malloc(device, state.actions);
        malloc(device, state.reset_mask);
    }
    template <typename CONFIG>
    void free(web::DEVICE& device, web::State<CONFIG>& state) {
        free(device, state.reset_mask);
        free(device, state.actions);
        free(device, state.next_states);
        free(device, state.states);
        free(device, state.parameters);
        free(device, state.rng);
        free(device, state.pilot_buffer);
        free(device, state.pilot_state);
        free(device, state.pilot);
        free(device, state.environment);
    }
    template <typename CONFIG>
    void init(web::DEVICE& device, web::State<CONFIG>& state, const std::string& scene_path, const std::vector<web::Drone>& entries) {
        using namespace web;
        using PARAMETER_SPEC = typename CONFIG::DYNAMICS_STATIC_PARAMETERS::PARAMETERS_SPEC;
        namespace models = rl::environments::l2f::parameters::dynamics;
        for(const auto& entry: entries) {
            state.environment.shared.drones.push_back({entry.path, {}});
            auto& drone = state.environment.shared.drones.back().parameters;
            auto& dynamics = drone.dynamics.dynamics;
            switch(entry.dynamics_index) {
                case 0:
                    dynamics = models::registry<models::REGISTRY::crazyflie, PARAMETER_SPEC>;
                    break;
                case 1:
                    dynamics = models::registry<models::REGISTRY::crazyflie_brushless, PARAMETER_SPEC>;
                    break;
                case 2:
                    dynamics = models::registry<models::REGISTRY::savagebee_pusher, PARAMETER_SPEC>;
                    break;
                case 3:
                    dynamics = models::registry<models::REGISTRY::arpl, PARAMETER_SPEC>;
                    break;
                case 4:
                    dynamics = models::registry<models::REGISTRY::x500, PARAMETER_SPEC>;
                    break;
                case 5:
                    dynamics = models::registry<models::REGISTRY::soft, PARAMETER_SPEC>;
                    break;
            }
            drone.cameras[0].fov = 80;
            for(TI dim = 0; dim < 3; dim++) {
                drone.cameras[0].mount.pose.position[dim] = entry.fpv_position[dim];
            }
            drone.cameras[0].mount.pose.orientation[0] = math::cos(device.math, (T)0.15);
            drone.cameras[0].mount.pose.orientation[2] = math::sin(device.math, (T)0.15);
            drone.cameras[1].fov = 50;
            drone.cameras[1].mount.pose.position[0] = -1.2;
            drone.cameras[1].mount.pose.position[2] = 0.6;
            const T half_pitch = math::atan2(device.math, (T)0.4, (T)1.2) / 2;
            drone.cameras[1].mount.pose.orientation[0] = math::cos(device.math, half_pitch);
            drone.cameras[1].mount.pose.orientation[2] = math::sin(device.math, half_pitch);
        }
        rendering::datasets::procthor::GLB dataset{{}, {scene_path}};
        init(device, state.environment, dataset);
        const auto& world = state.environment.environments[0];
        state.spawn = world.slots[world.active_slot].annotations.positions[0];
    }
    template <typename CONFIG>
    bool reset(web::DEVICE& device, web::State<CONFIG>& state) {
        using namespace web;
        auto& world = state.environment.environments[0];
        auto& parameters = get_ref(device, state.parameters, (TI)0);
        auto& physical = get_ref(device, state.states, (TI)0);
        init(device, state.rng, state.seed);
        initial_parameters(device, world, parameters);
        const auto& spawn = state.spawn;
        for(TI dim = 0; dim < 3; dim++) {
            parameters.scene_translation[dim] = spawn.position[dim];
        }
        parameters.scene_yaw_cos = math::cos(device.math, (T)spawn.yaw);
        parameters.scene_yaw_sin = math::sin(device.math, (T)spawn.yaw);
        parameters.shutter_fraction = CONFIG::SHUTTER_FRACTION_MIN;
        for(TI agent_i = 0; agent_i < CONFIG::N_AGENTS; agent_i++) {
            auto& agent = parameters.agents[agent_i];
            agent.drone = agent_i == 0 ? 0 : random::uniform_int_distribution(device.random, (TI)0, world.num_drones - 1, state.rng);
            static_cast<rl::environments::hyperdrone::DroneParameters<CONFIG>&>(agent) = state.environment.shared.drones[agent.drone].parameters;
            auto& agent_state = rl::environments::hyperdrone::_agent_state<CONFIG>(physical, agent_i);
            initial_state(device, world.dynamics, agent.dynamics, agent_state);
            auto& target = state.targets[agent_i];
            target = {};
            if(agent_i > 0) {
                bool found = false;
                const TI attempts = state.other_spawns.empty() ? SPAWN_ATTEMPTS : 1;
                for(TI attempt = 0; attempt < attempts && !found; attempt++) {
                    const auto candidate = state.other_spawns.empty()
                        ? rendering::datasets::annotations::sample_free_position(device, world.slots[world.active_slot].annotations, state.rng)
                        : state.other_spawns[agent_i - 1];
                    T offset[3];
                    for(TI dim = 0; dim < 3; dim++) {
                        offset[dim] = candidate.position[dim] - spawn.position[dim];
                    }
                    rl::environments::hyperdrone::rotate_scene_yaw(offset, target.position, parameters.scene_yaw_cos, -parameters.scene_yaw_sin);
                    found = true;
                    for(TI previous = 0; previous < agent_i; previous++) {
                        T distance_squared = 0;
                        for(TI dim = 0; dim < 3; dim++) {
                            const T delta = target.position[dim] - state.targets[previous].position[dim];
                            distance_squared += delta * delta;
                        }
                        if(distance_squared < DRONE_SEPARATION * DRONE_SEPARATION) {
                            found = false;
                        }
                    }
                    target.yaw = candidate.yaw - spawn.yaw;
                }
                if(!found) {
                    return false;
                }
                for(TI dim = 0; dim < 3; dim++) {
                    agent_state.position[dim] = target.position[dim];
                }
                agent_state.orientation[0] = math::cos(device.math, target.yaw / (T)2);
                agent_state.orientation[1] = 0;
                agent_state.orientation[2] = 0;
                agent_state.orientation[3] = math::sin(device.math, target.yaw / (T)2);
            }
        }
        reset(device, state.pilot, state.pilot_state, state.rng);
        state.moving = false;
        state.moving_vertical = false;
        set_all(device, state.reset_mask, true);
        set_all(device, state.actions, (T)0);
        request_render(device, world, state.reset_mask);
        return true;
    }
    template <typename CONFIG>
    void control(web::DEVICE& device, web::State<CONFIG>& state, web::T forward, web::T left, web::T up, web::T yaw_delta, web::T yaw_axis) {
        using namespace web;
        auto& physical = rl::environments::hyperdrone::_agent_state<CONFIG>(get_ref(device, state.states, (TI)0), (TI)0);
        auto& target = state.targets[0];
        const T dt = get_ref(device, state.parameters, (TI)0).agents[0].dynamics.integration.dt;
        update_yaw_target(device, state.pilot, physical, target, yaw_delta + yaw_axis * state.yaw_rate_limit * dt, dt, state.yaw_rate_limit, state.yaw_error_limit);
        const T norm = math::sqrt(device.math, forward * forward + left * left + up * up);
        const T scale = norm > 1 ? 1 / norm : 1;
        const bool moving = forward != 0 || left != 0;
        if(moving) {
            const auto& q = physical.orientation;
            const T heading = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
            const T c = math::cos(device.math, heading), s = math::sin(device.math, heading);
            const T direction[2] = {(c * forward - s * left) * scale, (s * forward + c * left) * scale};
            for(TI dim = 0; dim < 2; dim++) {
                target.position[dim] = physical.position[dim] + state.position_offset * direction[dim];
                target.linear_velocity[dim] = state.velocity_target * direction[dim];
            }
        } else {
            for(TI dim = 0; dim < 2; dim++) {
                if(state.moving) {
                    target.position[dim] = physical.position[dim];
                }
                target.linear_velocity[dim] = 0;
            }
        }
        state.moving = moving;
        if(up != 0) {
            target.position[2] = physical.position[2] + state.position_offset * up * scale;
            target.linear_velocity[2] = state.velocity_target * up * scale;
        } else {
            if(state.moving_vertical) {
                target.position[2] = physical.position[2];
            }
            target.linear_velocity[2] = 0;
        }
        state.moving_vertical = up != 0;
        for(TI agent_i = 0; agent_i < CONFIG::N_AGENTS; agent_i++) {
            const auto& agent_state = rl::environments::hyperdrone::_agent_state<CONFIG>(get_ref(device, state.states, (TI)0), agent_i);
            observe(device, state.pilot, state.pilot_state, state.pilot_buffer, agent_state, state.targets[agent_i], agent_i);
        }
        evaluate_step(device, state.pilot, state.pilot_state, state.pilot_buffer, state.rng);
        for(TI agent_i = 0; agent_i < CONFIG::N_AGENTS; agent_i++) {
            for(TI motor = 0; motor < 4; motor++) {
                set(device, state.actions, get(device, state.pilot_state.previous_actions, agent_i, motor), (TI)0, agent_i * 4 + motor);
            }
        }
    }
    template <typename CONFIG>
    void step(web::DEVICE& device, web::State<CONFIG>& state, web::T forward, web::T left, web::T up, web::T yaw_delta, web::T yaw_axis, web::TI steps) {
        using namespace web;
        auto& world = state.environment.environments[0];
        auto actions = matrix_view(device, state.actions);
        for(TI step_i = 0; step_i < steps; step_i++) {
            control(device, state, forward, left, up, yaw_delta / steps, yaw_axis);
            // Only the final control tick advances camera/flow history; scalar steps integrate dynamics alone.
            if(step_i + 1 < steps) {
                step(device, world, get_ref(device, state.parameters, (TI)0), get_ref(device, state.states, (TI)0), actions, get_ref(device, state.next_states, (TI)0), state.rng);
            } else {
                step(device, state.environment, state.parameters, state.states, state.actions, state.next_states, state.rng);
            }
            copy(device, device, state.next_states, state.states);
        }
    }
    template <typename CONFIG>
    void render(web::DEVICE& device, web::State<CONFIG>& state, web::Buffer<CONFIG>& buffer, rendering::raytracing::backends::webgpu::Canvas (&canvases)[web::VIEWS]) {
        using namespace web;
        namespace outputs = rendering::raytracing::outputs;
        render_launch(device, state.environment, state.parameters, state.states, state.reset_mask);
        set_all(device, state.reset_mask, false);
        auto& renderer = get_renderer(device, state.environment.environments[0]);
        present(device, renderer, canvases[0], outputs::RGB<>{}, (TI)0);
        present(device, renderer, canvases[1], outputs::RGB<>{}, (TI)1);
        present(device, renderer, canvases[2], outputs::Depth<outputs::format::RGBA8>{}, (TI)0);
        present(device, renderer, canvases[3], outputs::Segmentation<outputs::format::RGBA8, outputs::labels::Semantic>{}, (TI)0);
        present(device, renderer, canvases[4], outputs::Normals<outputs::format::RGBA8>{}, (TI)0);
        present(device, renderer, canvases[5], outputs::Flow<outputs::direction::Backward, outputs::format::RGBA8>{}, (TI)0);
        auto& segmentation = output(device, renderer, outputs::Segmentation<outputs::format::UInt32, outputs::labels::Semantic>{});
        Tensor<typename utils::typing::remove_reference_t<decltype(segmentation)>::SPEC> alias;
        alias._data = buffer.segmentation.data();
        copy(renderer.device, device, segmentation, alias);
    }
}

static void status(const char* message) {
    EM_ASM({
        if(Module.rlt_status) {
            Module.rlt_status(UTF8ToString($0));
        }
    }, message);
}
static void show_gpu_info(WGPUAdapter adapter) {
    WGPUAdapterInfo info{};
    if(wgpuAdapterGetInfo(adapter, &info) != WGPUStatus_Success) {
        return;
    }
    const auto string = [](WGPUStringView value) {
        const size_t length = value.data == nullptr ? 0 : (value.length == WGPU_STRLEN ? std::strlen(value.data) : value.length);
        return std::string(value.data == nullptr ? "" : value.data, length);
    };
    const std::string json = nlohmann::json{
        {"vendor", string(info.vendor)}, {"architecture", string(info.architecture)},
        {"device", string(info.device)}, {"description", string(info.description)}
    }.dump();
    wgpuAdapterInfoFreeMembers(info);
    EM_ASM({
        Module.rlt_gpu_info(JSON.parse(UTF8ToString($0)));
    }, json.c_str());
}
static void read_input(float input[14]) {
    EM_ASM({
        const input = Module.rlt_input();
        for(let i = 0; i < 14; i++) {
            HEAPF32[($0 >> 2) + i] = input[i];
        }
    }, input);
}
template <typename CONFIG>
static void present(web::DEVICE& device, const web::State<CONFIG>& state, const web::Buffer<CONFIG>& buffer) {
    using namespace rl_tools;
    using rl::environments::hyperdrone::rotate_scene_yaw;
    using rl::environments::l2f::quaternion_multiplication;
    const auto& physical = rl::environments::hyperdrone::_agent_state<CONFIG>(get_ref(device, state.states, (web::TI)0), (web::TI)0);
    const auto& parameters = get_ref(device, state.parameters, (web::TI)0);
    web::T position[3], orientation[4];
    rotate_scene_yaw(physical.position, position, parameters.scene_yaw_cos, parameters.scene_yaw_sin);
    for(web::TI dim = 0; dim < 3; dim++) {
        position[dim] += parameters.scene_translation[dim];
    }
    const web::T half_yaw = math::atan2(device.math, parameters.scene_yaw_sin, parameters.scene_yaw_cos) / 2;
    const web::T scene_orientation[4] = {math::cos(device.math, half_yaw), 0, 0, math::sin(device.math, half_yaw)};
    quaternion_multiplication<web::DEVICE, web::T>(scene_orientation, physical.orientation, orientation);
    EM_ASM({
        if(Module.rlt_present) {
            Module.rlt_present($0, $1, $2, $3, $4, $5, $6, $7, $8, $9);
        }
    }, buffer.segmentation.data(), (int)CONFIG::CAM_WIDTH, (int)CONFIG::CAM_HEIGHT,
        position[0], position[1], position[2], orientation[0], orientation[1], orientation[2], orientation[3]);
}

template <typename CONFIG>
static std::string configuration_id() {
    return std::string(CONFIG::FIDELITY) + "-mb" + std::to_string(CONFIG::MOTION_BLUR_SAMPLES)
        + "-aa" + std::to_string(CONFIG::ANTI_ALIASING_GRID_SIZE)
        + "-" + std::to_string(CONFIG::CAM_WIDTH) + "x" + std::to_string(CONFIG::CAM_HEIGHT);
}

template <typename... CONFIGS>
static std::string configuration_catalog(web::Configurations<CONFIGS...>) {
    auto catalog = nlohmann::json::array();
    (catalog.push_back({
        {"id", configuration_id<CONFIGS>()},
        {"fidelity", CONFIGS::FIDELITY},
        {"motion_blur", CONFIGS::ENABLE_MOTION_BLUR},
        {"antialiasing", CONFIGS::ENABLE_ANTI_ALIASING},
        {"width", CONFIGS::CAM_WIDTH},
        {"height", CONFIGS::CAM_HEIGHT},
        {"num_agents", CONFIGS::N_AGENTS}
    }), ...);
    return catalog.dump();
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_api_version() {
    return 10;
}

extern "C" EMSCRIPTEN_KEEPALIVE const char* demo_configurations() {
    static const std::string catalog = configuration_catalog(web::CompiledConfigurations{});
    return catalog.c_str();
}

extern "C" EMSCRIPTEN_KEEPALIVE const char* demo_scenes() {
    static const std::string catalog = [] {
        auto scenes = nlohmann::json::array();
        for(const auto& scene: rl_tools::rendering::datasets::procthor::Train200::SCENES) {
            scenes.push_back({{"name", scene.id}, {"reference", scene.reference}});
        }
        return scenes.dump();
    }();
    return catalog.c_str();
}

template <typename CONFIG>
static int run(const char* scene_path, const char* policy_path, int max_frames, const std::vector<web::Drone>& drones, uint32_t seed, const nlohmann::json& placement) {
    using namespace rl_tools;
    web::DEVICE device;
    init(device);
    web::State<CONFIG> state;
    web::Buffer<CONFIG> buffer;
    rendering::raytracing::backends::webgpu::Canvas canvases[web::VIEWS];
    malloc(device, state);
    status("Loading RAPTOR ...");
    if(!load(device, state.pilot, std::string(policy_path))) {
        status("The RAPTOR checkpoint could not be loaded");
        free(device, state);
        return 1;
    }
    status("Loading the scene and finding a clear starting position ...");
    state.seed = seed;
    init(device, state, scene_path, drones);
    show_gpu_info(get_renderer(device, state.environment.environments[0]).backend->adapter);
    auto classes = nlohmann::json::object();
    for(const auto& entry: state.environment.shared.segmentation_taxonomy->classes) {
        classes[std::to_string(entry.first)] = entry.second;
    }
    const std::string class_names = classes.dump();
    EM_ASM({
        Module.rlt_segmentation_classes(JSON.parse(UTF8ToString($0)));
    }, class_names.c_str());
    const auto& rig = state.environment.shared.drone_bodies[0].rig;
    EM_ASM({
        Module.rlt_motor_layout($0, $1, $2);
    }, rig.rotor_position_deviation, rig.wheelbase, rl::environments::hyperdrone::rig::ROTOR_POSITION_WARNING_THRESHOLD);
    if(placement.contains("position")) {
        for(web::TI dim = 0; dim < 3; dim++) {
            state.spawn.position[dim] = placement.at("position").at(dim).get<web::T>();
        }
        state.spawn.yaw = placement.at("yaw").get<web::T>();
    }
    if(placement.contains("other_positions")) {
        for(const auto& position: placement.at("other_positions")) {
            auto spawn = state.spawn;
            for(web::TI dim = 0; dim < 3; dim++) {
                spawn.position[dim] = position.at(dim).get<web::T>();
            }
            state.other_spawns.push_back(spawn);
        }
    }
    for(web::TI view_i = 0; view_i < web::VIEWS; view_i++) {
        init(device, get_renderer(device, state.environment.environments[0]), canvases[view_i], web::CANVASES[view_i]);
    }
    const auto release = [&]() {
        for(auto& canvas: canvases) {
            free(device, canvas);
        }
        free(device, state);
    };
    if(!reset(device, state)) {
        status("Could not place seven separated drones in this scene");
        release();
        return 3;
    }
    status("Simulation running — click a camera to take control");
    const double dt = get_ref(device, state.parameters, (web::TI)0).agents[0].dynamics.integration.dt;
    double previous_ms = emscripten_get_now();
    double accumulator = 0;
    bool was_active = false;
    for(int frame = 0; max_frames <= 0 || frame < max_frames;) {
        float input[14] = {};
        read_input(input);
        const double frame_period = max_frames > 0 ? web::FRAME_PERIOD : 1.0 / std::clamp((double)input[10], 10.0, 240.0);
        state.yaw_rate_limit = input[6];
        state.yaw_error_limit = input[7];
        state.position_offset = input[8];
        state.velocity_target = input[9];
        const bool active = max_frames > 0 || input[5] != 0;
        if(input[4] != 0) {
            if(!reset(device, state)) {
                release();
                return 3;
            }
            accumulator = 0;
        }
        const double now_ms = emscripten_get_now();
        if(active && was_active && input[4] == 0) {
            accumulator += max_frames > 0 ? web::FRAME_PERIOD : std::min((now_ms - previous_ms) / 1000.0, 0.1);
        }
        previous_ms = now_ms;
        if(!active && was_active) {
            const auto& physical = rl::environments::hyperdrone::_agent_state<CONFIG>(get_ref(device, state.states, (web::TI)0), (web::TI)0);
            for(web::TI dim = 0; dim < 3; dim++) {
                state.targets[0].position[dim] = physical.position[dim];
                state.targets[0].linear_velocity[dim] = 0;
            }
            state.moving = false;
            state.moving_vertical = false;
            accumulator = 0;
        }
        was_active = active;
        if(active) {
            const auto steps = (web::TI)(accumulator / dt);
            step(device, state, input[0], input[1], input[2], input[3], input[11], steps);
            accumulator -= steps * dt;
        }
        if(frame == 0 || active || input[4] != 0) {
            auto& cameras = get_ref(device, state.parameters, (web::TI)0).agents[0].cameras;
            for(web::TI camera_i = 0; camera_i < 2; camera_i++) {
                const web::T fov = std::clamp((web::T)input[12 + camera_i], (web::T)30, (web::T)140);
                if(cameras[camera_i].fov != fov) {
                    cameras[camera_i].fov = fov;
                    request_render(device, state.environment.environments[0]);
                }
            }
            render(device, state, buffer, canvases);
            present(device, state, buffer);
            frame++;
        }
        const double elapsed_ms = emscripten_get_now() - now_ms;
        const int sleep_ms = active ? (int)std::max(1.0, frame_period * 1000.0 - elapsed_ms) : 16;
        emscripten_sleep(sleep_ms);
    }
    release();
    return 0;
}

template <typename... CONFIGS>
static int dispatch_run(web::Configurations<CONFIGS...>, const std::string& selected, web::TI num_agents, const char* scene_path, const char* policy_path, int max_frames, const std::vector<web::Drone>& drones, uint32_t seed, const nlohmann::json& placement) {
    int result = 2;
    const bool found = ((selected == configuration_id<CONFIGS>() && num_agents == CONFIGS::N_AGENTS
        ? (result = run<CONFIGS>(scene_path, policy_path, max_frames, drones, seed, placement), true)
        : false) || ...);
    if(!found) {
        status("The selected rendering and drone configuration is not included in this build");
    }
    return result;
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_run(const char* scene_path, const char* policy_path, int max_frames, const char* session_json, const char* selected_configuration) {
    const auto session = nlohmann::json::parse(session_json, nullptr, false);
    if(!session.is_object() || !session.contains("num_agents") || !session.at("num_agents").is_number_integer()
        || (session.at("num_agents") != 1 && session.at("num_agents") != 8)
        || !session.contains("seed") || !session.at("seed").is_number_unsigned() || session.at("seed") > uint64_t{4294967295}
        || !session.contains("drones") || !session.at("drones").is_array()) {
        status("The flight settings are invalid");
        return 1;
    }
    const auto num_agents = session.at("num_agents").get<web::TI>();
    const auto& entries = session.at("drones");
    if(entries.size() != (num_agents == 1 ? 1 : 6)) {
        status("The flight needs one airframe alone, or all six airframes with other drones");
        return 1;
    }
    bool models[6] = {};
    std::vector<web::Drone> drones;
    for(const auto& entry: entries) {
        if(!entry.is_object() || !entry.contains("path") || !entry.at("path").is_string() || entry.at("path").get<std::string>().empty()
            || !entry.contains("dynamics_index") || !entry.at("dynamics_index").is_number_integer() || entry.at("dynamics_index") < 0 || entry.at("dynamics_index") > 5
            || !entry.contains("fpv_position") || !entry.at("fpv_position").is_array() || entry.at("fpv_position").size() != 3) {
            status("The airframe registry is invalid");
            return 1;
        }
        web::Drone drone{entry.at("path").get<std::string>(), entry.at("dynamics_index").get<int>(), {}};
        if(models[drone.dynamics_index]) {
            status("The airframe registry contains a repeated dynamics model");
            return 1;
        }
        models[drone.dynamics_index] = true;
        for(web::TI dim = 0; dim < 3; dim++) {
            const auto& value = entry.at("fpv_position").at(dim);
            if(!value.is_number() || !rl_tools::rendering::rig::detail::is_finite(drone.fpv_position[dim] = value.get<web::T>())) {
                status("The airframe camera position is invalid");
                return 1;
            }
        }
        drones.push_back(drone);
    }
    if(session.contains("position")) {
        if(!session.at("position").is_array() || session.at("position").size() != 3
            || !session.contains("yaw") || !session.at("yaw").is_number()
            || !rl_tools::rendering::rig::detail::is_finite(session.at("yaw").get<web::T>())) {
            status("The starting pose is invalid");
            return 1;
        }
        for(const auto& value: session.at("position")) {
            if(!value.is_number() || !rl_tools::rendering::rig::detail::is_finite(value.get<web::T>())) {
                status("The starting pose is invalid");
                return 1;
            }
        }
    }
    if(session.contains("other_positions")) {
        const auto& positions = session.at("other_positions");
        if(!positions.is_array() || positions.size() != num_agents - 1) {
            status("The flight needs one starting position per other drone");
            return 1;
        }
        for(const auto& position: positions) {
            if(!position.is_array() || position.size() != 3) {
                status("The other drone starting positions are invalid");
                return 1;
            }
            for(const auto& value: position) {
                if(!value.is_number() || !rl_tools::rendering::rig::detail::is_finite(value.get<web::T>())) {
                    status("The other drone starting positions are invalid");
                    return 1;
                }
            }
        }
    }
    return dispatch_run(web::CompiledConfigurations{}, selected_configuration == nullptr ? "" : selected_configuration,
        num_agents, scene_path, policy_path, max_frames, drones, session.at("seed").get<uint32_t>(), session);
}

int main() {
    return 0;
}
