#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/raptor/operations_cpu.h>
#include <rl_tools/rendering/datasets/procthor/catalog.h>

#include <emscripten.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace rl_tools::rl::environments::hyperdrone::web {
    using DEVICE = devices::DEVICE_FACTORY<>;
    using TI = DEVICE::index_t;
    using T = float;
    using TYPE_POLICY = numeric_types::Policy<T>;
    using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
    template <typename T_SHADING, TI T_MOTION_BLUR_SAMPLES, TI T_ANTI_ALIASING_GRID_SIZE, TI T_RESOLUTION_SCALE>
    struct Config: presets::X500FPV<T, TI> {
        using SHADING = T_SHADING;
        static_assert(utils::typing::is_same_v<SHADING, rendering::raytracing::Low>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::Medium>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::High>
            || utils::typing::is_same_v<SHADING, rendering::raytracing::VeryHigh>);
        static_assert(T_MOTION_BLUR_SAMPLES == 1 || T_MOTION_BLUR_SAMPLES == 4);
        static_assert(T_ANTI_ALIASING_GRID_SIZE == 1 || T_ANTI_ALIASING_GRID_SIZE == 2);
        static_assert(T_RESOLUTION_SCALE == 1 || T_RESOLUTION_SCALE == 2);
        static constexpr const char* FIDELITY = utils::typing::is_same_v<SHADING, rendering::raytracing::Low> ? "low"
            : utils::typing::is_same_v<SHADING, rendering::raytracing::Medium> ? "medium"
            : utils::typing::is_same_v<SHADING, rendering::raytracing::High> ? "high" : "very-high";
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
        static constexpr TI N_CAMERAS_PER_AGENT = 2;
        static constexpr TI CAM_WIDTH = 480 * T_RESOLUTION_SCALE;
        static constexpr TI CAM_HEIGHT = 270 * T_RESOLUTION_SCALE;
        static constexpr TI NUM_PROBES = 64;
        static constexpr bool OUTPUT_DEPTH = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
        static constexpr bool SEMANTIC_SEGMENTATION = true;
        static constexpr bool OUTPUT_NORMALS = true;
        static constexpr bool OUTPUT_FLOW = true;
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
    using PILOT = Raptor<raptor::Specification<TYPE_POLICY, TI, 1>>;
    constexpr TI VIEWS = 6;
    constexpr T DEPTH_DISPLAY_METERS = 10;
    constexpr T FLOW_DISPLAY_PIXELS = 20;
    constexpr double FRAME_PERIOD = 1.0 / 60.0;

    template <typename CONFIG>
    struct State {
        using WORLD = World<CONFIG>;
        using ENVIRONMENT = MultiEnvironment<WORLD, 1>;
        ENVIRONMENT environment;
        PILOT pilot;
        RNG rng;
        Tensor<tensor::Specification<typename WORLD::Parameters, TI, tensor::Shape<TI, 1>>> parameters;
        Tensor<tensor::Specification<typename WORLD::State, TI, tensor::Shape<TI, 1>>> states, next_states;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, 4>>> actions;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 1>>> reset_mask;
        raptor::Target<T> target;
        rendering::datasets::annotations::FreeSpacePosition<T> spawn;
        T position_offset = 0.3;
        T velocity_target = 0.7;
        T yaw_rate_limit = PILOT::SPEC::YAW_RATE_LIMIT;
        T yaw_error_limit = PILOT::SPEC::YAW_ERROR_LIMIT;
        T previous_actions[4] = {};
        TI motor_order[4] = {};
        bool moving = false;
        bool moving_vertical = false;
    };
    template <typename CONFIG>
    struct Buffer {
        static constexpr TI PIXELS = CONFIG::CAM_WIDTH * CONFIG::CAM_HEIGHT;
        std::vector<uint32_t> rgba = std::vector<uint32_t>(VIEWS * PIXELS);
        std::vector<float> depth = std::vector<float>(PIXELS);
        std::vector<uint32_t> segmentation = std::vector<uint32_t>(PIXELS);
        std::vector<float> normals = std::vector<float>(PIXELS * 3);
        std::vector<float> flow = std::vector<float>(PIXELS * 2);
        std::vector<uint32_t> flow_status = std::vector<uint32_t>(PIXELS);
    };
}

#include <hyperdrone_web_configurations.h>

namespace web = rl_tools::rl::environments::hyperdrone::web;

namespace rl_tools {
    template <typename CONFIG>
    void malloc(web::DEVICE& device, web::State<CONFIG>& state) {
        malloc(device, state.environment);
        malloc(device, state.pilot);
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
        free(device, state.pilot);
        free(device, state.environment);
    }
    template <typename CONFIG>
    void init(web::DEVICE& device, web::State<CONFIG>& state, const std::string& scene_path, const std::string& drone_path, int drone_index, const web::T fpv_position[3]) {
        using namespace web;
        using PARAMETER_SPEC = typename CONFIG::DYNAMICS_STATIC_PARAMETERS::PARAMETERS_SPEC;
        namespace models = rl::environments::l2f::parameters::dynamics;
        state.environment.shared.drones.push_back({drone_path, {}});
        auto& drone = state.environment.shared.drones[0].parameters;
        auto& dynamics = drone.dynamics.dynamics;
        switch(drone_index) {
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
            default:
                dynamics = models::registry<models::REGISTRY::crazyflie, PARAMETER_SPEC>;
                break;
        }
        const auto policy_dynamics = models::registry<models::REGISTRY::x500, PARAMETER_SPEC>;
        for(TI motor = 0; motor < 4; motor++) {
            TI policy_motor = 0;
            for(; policy_motor < 4; policy_motor++) {
                if((dynamics.rotor_positions[motor][0] > 0) == (policy_dynamics.rotor_positions[policy_motor][0] > 0)
                    && (dynamics.rotor_positions[motor][1] > 0) == (policy_dynamics.rotor_positions[policy_motor][1] > 0)) {
                    break;
                }
            }
            utils::assert_exit(device, policy_motor < 4, "HyperDrone: unsupported rotor layout");
            state.motor_order[motor] = policy_motor;
        }
        constexpr T RPM_SCALE = 10000;
        dynamics.action_limit.min *= RPM_SCALE;
        dynamics.action_limit.max *= RPM_SCALE;
        for(TI motor = 0; motor < 4; motor++) {
            dynamics.rotor_thrust_coefficients[motor][1] /= RPM_SCALE;
            dynamics.rotor_thrust_coefficients[motor][2] /= RPM_SCALE * RPM_SCALE;
        }
        drone.cameras[0].fov = 80;
        for(TI dim = 0; dim < 3; dim++) {
            drone.cameras[0].mount.pose.position[dim] = fpv_position[dim];
        }
        drone.cameras[0].mount.pose.orientation[0] = math::cos(device.math, (T)0.15);
        drone.cameras[0].mount.pose.orientation[2] = math::sin(device.math, (T)0.15);
        drone.cameras[1].fov = 50;
        drone.cameras[1].mount.pose.position[0] = -1.2;
        drone.cameras[1].mount.pose.position[2] = 0.4;
        const T half_pitch = math::atan2(device.math, (T)0.4, (T)1.2) / 2;
        drone.cameras[1].mount.pose.orientation[0] = math::cos(device.math, half_pitch);
        drone.cameras[1].mount.pose.orientation[2] = math::sin(device.math, half_pitch);
        rendering::datasets::procthor::GLB dataset{{}, {scene_path}};
        init(device, state.environment, dataset);
        const auto& world = state.environment.environments[0];
        state.spawn = world.slots[world.active_slot].annotations.positions[0];
        init(device, state.rng, 1337);
    }
    template <typename CONFIG>
    void reset(web::DEVICE& device, web::State<CONFIG>& state) {
        using namespace web;
        auto& world = state.environment.environments[0];
        auto& parameters = get_ref(device, state.parameters, (TI)0);
        initial_parameters(device, world, parameters);
        const auto& spawn = state.spawn;
        for(TI dim = 0; dim < 3; dim++) {
            parameters.scene_translation[dim] = spawn.position[dim];
        }
        parameters.scene_yaw_cos = math::cos(device.math, (T)spawn.yaw);
        parameters.scene_yaw_sin = math::sin(device.math, (T)spawn.yaw);
        parameters.shutter_fraction = CONFIG::SHUTTER_FRACTION_MIN;
        initial_state(device, world.dynamics, parameters.agents[0].dynamics, get_ref(device, state.states, (TI)0));
        reset(device, state.pilot, state.rng);
        state.target = {};
        state.moving = false;
        state.moving_vertical = false;
        for(TI motor = 0; motor < 4; motor++) {
            state.previous_actions[motor] = 0;
        }
        set_all(device, state.reset_mask, true);
        set_all(device, state.actions, (T)0);
        request_render(device, world, state.reset_mask);
    }
    template <typename CONFIG>
    void control(web::DEVICE& device, web::State<CONFIG>& state, web::T forward, web::T left, web::T up, web::T yaw_delta, web::T yaw_axis) {
        using namespace web;
        auto& physical = get_ref(device, state.states, (TI)0);
        const T dt = get_ref(device, state.parameters, (TI)0).agents[0].dynamics.integration.dt;
        update_yaw_target(device, state.pilot, physical, state.target, yaw_delta + yaw_axis * state.yaw_rate_limit * dt, dt, state.yaw_rate_limit, state.yaw_error_limit);
        const T norm = math::sqrt(device.math, forward * forward + left * left + up * up);
        const T scale = norm > 1 ? 1 / norm : 1;
        const bool moving = forward != 0 || left != 0;
        if(moving) {
            const auto& q = physical.orientation;
            const T heading = math::atan2(device.math, (T)2 * (q[0] * q[3] + q[1] * q[2]), (T)1 - (T)2 * (q[2] * q[2] + q[3] * q[3]));
            const T c = math::cos(device.math, heading), s = math::sin(device.math, heading);
            const T direction[2] = {(c * forward - s * left) * scale, (s * forward + c * left) * scale};
            for(TI dim = 0; dim < 2; dim++) {
                state.target.position[dim] = physical.position[dim] + state.position_offset * direction[dim];
                state.target.linear_velocity[dim] = state.velocity_target * direction[dim];
            }
        } else {
            for(TI dim = 0; dim < 2; dim++) {
                if(state.moving) {
                    state.target.position[dim] = physical.position[dim];
                }
                state.target.linear_velocity[dim] = 0;
            }
        }
        state.moving = moving;
        if(up != 0) {
            state.target.position[2] = physical.position[2] + state.position_offset * up * scale;
            state.target.linear_velocity[2] = state.velocity_target * up * scale;
        } else {
            if(state.moving_vertical) {
                state.target.position[2] = physical.position[2];
            }
            state.target.linear_velocity[2] = 0;
        }
        state.moving_vertical = up != 0;
        observe(device, state.pilot, physical, state.target, state.previous_actions, (TI)0);
        evaluate_step(device, state.pilot, state.rng);
        for(TI motor = 0; motor < 4; motor++) {
            state.previous_actions[motor] = get(device, state.pilot.actions, (TI)0, motor);
        }
        for(TI motor = 0; motor < 4; motor++) {
            set(device, state.actions, state.previous_actions[state.motor_order[motor]], (TI)0, motor);
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
    template <typename RENDERER, typename TENSOR, typename T>
    void copy_onboard(web::DEVICE& device, RENDERER& renderer, TENSOR& source, T* destination) {
        auto onboard = view(device, source, (web::TI)0);
        Tensor<typename decltype(onboard)::SPEC> alias;
        alias._data = destination;
        copy(renderer.device, device, onboard, alias);
    }
    template <typename CONFIG>
    void render(web::DEVICE& device, web::State<CONFIG>& state, web::Buffer<CONFIG>& buffer) {
        using namespace web;
        using WORLD = typename State<CONFIG>::WORLD;
        constexpr TI PIXELS = Buffer<CONFIG>::PIXELS;
        render(device, state.environment, state.parameters, state.states, state.reset_mask);
        set_all(device, state.reset_mask, false);
        auto& renderer = get_renderer(device, state.environment.environments[0]);
        auto& rgb = frame_buffer(device, renderer);
        Tensor<typename WORLD::RENDERER::FB_TENSOR_SPEC> alias;
        alias._data = buffer.rgba.data();
        copy(renderer.device, device, rgb, alias);
        copy_onboard(device, renderer, depth_buffer(device, renderer), buffer.depth.data());
        copy_onboard(device, renderer, segmentation_buffer(device, renderer), buffer.segmentation.data());
        copy_onboard(device, renderer, normals_buffer(device, renderer), buffer.normals.data());
        copy_onboard(device, renderer, flow_buffer(device, renderer), buffer.flow.data());
        copy_onboard(device, renderer, flow_status_buffer(device, renderer), buffer.flow_status.data());
        for(TI pixel = 0; pixel < PIXELS; pixel++) {
            const auto shade = (uint32_t)(math::clamp(device.math, buffer.depth[pixel] / DEPTH_DISPLAY_METERS, (T)0, (T)1) * 255);
            buffer.rgba[2 * PIXELS + pixel] = 0xFF000000u | shade | (shade << 8) | (shade << 16);
            buffer.rgba[3 * PIXELS + pixel] = rendering::raytracing::detail::segmentation_id_to_rgba(buffer.segmentation[pixel]);
            buffer.rgba[4 * PIXELS + pixel] = rendering::raytracing::detail::normal_to_rgba(buffer.normals.data() + pixel * 3);
            const auto status = buffer.flow_status[pixel];
            buffer.rgba[5 * PIXELS + pixel] = status == rendering::raytracing::FlowStatus::VALID
                ? rendering::raytracing::detail::flow_to_rgba(buffer.flow[pixel * 2], buffer.flow[pixel * 2 + 1], FLOW_DISPLAY_PIXELS)
                : rendering::raytracing::detail::flow_status_to_rgba(status);
        }
    }
}

static void status(const char* message) {
    EM_ASM({
        if(Module.rlt_status) {
            Module.rlt_status(UTF8ToString($0));
        }
    }, message);
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
    const auto& physical = get_ref(device, state.states, (web::TI)0);
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
            Module.rlt_present($0, $1, $2, $3, $4, $5, $6, $7, $8, $9, $10);
        }
    }, buffer.rgba.data(), (int)CONFIG::CAM_WIDTH, (int)CONFIG::CAM_HEIGHT, (int)web::VIEWS,
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
        {"height", CONFIGS::CAM_HEIGHT}
    }), ...);
    return catalog.dump();
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_api_version() {
    return 5;
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
static int run(const char* scene_path, const char* drone_path, const char* policy_path, int drone_index, int max_frames, const char* pose_json) {
    using namespace rl_tools;
    const auto placement = nlohmann::json::parse(pose_json, nullptr, false);
    if(!placement.is_object() || !placement.contains("fpv_position") || !placement.at("fpv_position").is_array() || placement.at("fpv_position").size() != 3) {
        status("The airframe camera position is invalid");
        return 1;
    }
    web::T fpv_position[3];
    for(web::TI dim = 0; dim < 3; dim++) {
        const auto& value = placement.at("fpv_position").at(dim);
        if(!value.is_number() || !rendering::rig::detail::is_finite(fpv_position[dim] = value.get<web::T>())) {
            status("The airframe camera position is invalid");
            return 1;
        }
    }
    web::DEVICE device;
    init(device);
    web::State<CONFIG> state;
    web::Buffer<CONFIG> buffer;
    malloc(device, state);
    status("Loading RAPTOR ...");
    if(!load(device, state.pilot, std::string(policy_path))) {
        status("The RAPTOR checkpoint could not be loaded");
        free(device, state);
        return 1;
    }
    status("Loading the scene and finding a clear starting position ...");
    init(device, state, scene_path, drone_path, drone_index, fpv_position);
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
    reset(device, state);
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
            reset(device, state);
            accumulator = 0;
        }
        const double now_ms = emscripten_get_now();
        if(active && was_active) {
            accumulator += max_frames > 0 ? web::FRAME_PERIOD : std::min((now_ms - previous_ms) / 1000.0, 0.1);
        }
        previous_ms = now_ms;
        if(!active && was_active) {
            const auto& physical = get_ref(device, state.states, (web::TI)0);
            for(web::TI dim = 0; dim < 3; dim++) {
                state.target.position[dim] = physical.position[dim];
                state.target.linear_velocity[dim] = 0;
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
            render(device, state, buffer);
            present(device, state, buffer);
            frame++;
        }
        const double elapsed_ms = emscripten_get_now() - now_ms;
        const int sleep_ms = active ? (int)std::max(1.0, frame_period * 1000.0 - elapsed_ms) : 16;
        emscripten_sleep(sleep_ms);
    }
    free(device, state);
    return 0;
}

template <typename... CONFIGS>
static int dispatch_run(web::Configurations<CONFIGS...>, const std::string& selected, const char* scene_path, const char* drone_path, const char* policy_path, int drone_index, int max_frames, const char* pose_json) {
    int result = 2;
    const bool found = ((selected == configuration_id<CONFIGS>()
        ? (result = run<CONFIGS>(scene_path, drone_path, policy_path, drone_index, max_frames, pose_json), true)
        : false) || ...);
    if(!found) {
        status("The selected rendering configuration is not included in this build");
    }
    return result;
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_run(const char* scene_path, const char* drone_path, const char* policy_path, int drone_index, int max_frames, const char* pose_json, const char* selected_configuration) {
    return dispatch_run(web::CompiledConfigurations{}, selected_configuration == nullptr ? "" : selected_configuration,
        scene_path, drone_path, policy_path, drone_index, max_frames, pose_json);
}

int main() {
    return 0;
}
