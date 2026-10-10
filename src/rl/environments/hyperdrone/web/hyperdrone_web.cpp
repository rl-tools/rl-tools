#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rl/environments/hyperdrone/presets.h>
#include <rl_tools/rl/environments/hyperdrone/operations_cpu.h>
#include <rl_tools/rl/environments/hyperdrone/raptor/operations_cpu.h>

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
    struct Config: presets::X500FPV<T, TI> {
        static constexpr TI INSTANCES_PER_ENVIRONMENT = 1;
        static constexpr TI N_CAMERAS_PER_AGENT = 2;
        static constexpr TI CAM_WIDTH = 480;
        static constexpr TI CAM_HEIGHT = 270;
        static constexpr TI NUM_PROBES = 64;
        static constexpr bool OUTPUT_DEPTH = true;
        static constexpr bool OUTPUT_SEGMENTATION = true;
        static constexpr bool SEMANTIC_SEGMENTATION = true;
        static constexpr bool OUTPUT_NORMALS = true;
        static constexpr bool OUTPUT_FLOW = true;
        static constexpr bool ENABLE_MOTION_BLUR = true;
        static constexpr TI MOTION_BLUR_SAMPLES = 4;
        static constexpr T SHUTTER_FRACTION_MIN = 0.5;
        static constexpr T SHUTTER_FRACTION_MAX = 0.5;
    };
    using WORLD = World<Config>;
    using ENVIRONMENT = MultiEnvironment<WORLD, 1>;
    using PILOT = Raptor<raptor::Specification<TYPE_POLICY, TI, 1>>;
    constexpr TI PIXELS = Config::CAM_WIDTH * Config::CAM_HEIGHT;
    constexpr TI VIEWS = 6;
    constexpr T DEPTH_DISPLAY_METERS = 10;
    constexpr T FLOW_DISPLAY_PIXELS = 20;
    constexpr double FRAME_PERIOD = 1.0 / 60.0;

    struct State {
        ENVIRONMENT environment;
        PILOT pilot;
        RNG rng;
        Tensor<tensor::Specification<WORLD::Parameters, TI, tensor::Shape<TI, 1>>> parameters;
        Tensor<tensor::Specification<WORLD::State, TI, tensor::Shape<TI, 1>>> states, next_states;
        Tensor<tensor::Specification<T, TI, tensor::Shape<TI, 1, 4>>> actions;
        Tensor<tensor::Specification<bool, TI, tensor::Shape<TI, 1>>> reset_mask;
        raptor::Target<T> target;
        T position_offset = 0.3;
        T velocity_target = 0.7;
        T yaw_rate_limit = PILOT::SPEC::YAW_RATE_LIMIT;
        T yaw_error_limit = PILOT::SPEC::YAW_ERROR_LIMIT;
        T previous_actions[4] = {};
        TI motor_order[4] = {};
        bool moving = false;
    };
    struct Buffer {
        std::vector<uint32_t> rgba = std::vector<uint32_t>(VIEWS * PIXELS);
        std::vector<float> depth = std::vector<float>(PIXELS);
        std::vector<uint32_t> segmentation = std::vector<uint32_t>(PIXELS);
        std::vector<float> normals = std::vector<float>(PIXELS * 3);
        std::vector<float> flow = std::vector<float>(PIXELS * 2);
        std::vector<uint32_t> flow_status = std::vector<uint32_t>(PIXELS);
    };
}

namespace web = rl_tools::rl::environments::hyperdrone::web;

namespace rl_tools {
    void malloc(web::DEVICE& device, web::State& state) {
        malloc(device, state.environment);
        malloc(device, state.pilot);
        malloc(device, state.rng);
        malloc(device, state.parameters);
        malloc(device, state.states);
        malloc(device, state.next_states);
        malloc(device, state.actions);
        malloc(device, state.reset_mask);
    }
    void free(web::DEVICE& device, web::State& state) {
        free(device, state.reset_mask);
        free(device, state.actions);
        free(device, state.next_states);
        free(device, state.states);
        free(device, state.parameters);
        free(device, state.rng);
        free(device, state.pilot);
        free(device, state.environment);
    }
    void init(web::DEVICE& device, web::State& state, const std::string& scene_path, const std::string& drone_path, int drone_index) {
        using namespace web;
        using PARAMETER_SPEC = Config::DYNAMICS_STATIC_PARAMETERS::PARAMETERS_SPEC;
        namespace models = rl::environments::l2f::parameters::dynamics;
        state.environment.shared.drones.push_back({drone_path, {}});
        auto& drone = state.environment.shared.drones[0].parameters;
        auto& dynamics = drone.dynamics.dynamics;
        switch(drone_index) {
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
        drone.cameras[0].mount.pose.position[0] = 0.10;
        drone.cameras[0].mount.pose.position[2] = 0.32;
        drone.cameras[0].mount.pose.orientation[0] = math::cos(device.math, (T)0.15);
        drone.cameras[0].mount.pose.orientation[2] = math::sin(device.math, (T)0.15);
        drone.cameras[1].fov = 75;
        drone.cameras[1].mount.pose.position[0] = -1.2;
        drone.cameras[1].mount.pose.position[2] = 0.4;
        const T half_pitch = math::atan2(device.math, (T)0.4, (T)1.2) / 2;
        drone.cameras[1].mount.pose.orientation[0] = math::cos(device.math, half_pitch);
        drone.cameras[1].mount.pose.orientation[2] = math::sin(device.math, half_pitch);
        rendering::datasets::procthor::GLB dataset{{}, {scene_path}};
        init(device, state.environment, dataset);
        init(device, state.rng, 1337);
    }
    void reset(web::DEVICE& device, web::State& state) {
        using namespace web;
        auto& world = state.environment.environments[0];
        auto& parameters = get_ref(device, state.parameters, (TI)0);
        initial_parameters(device, world, parameters);
        const auto& spawn = world.slots[world.active_slot].annotations.positions[0];
        for(TI dim = 0; dim < 3; dim++) {
            parameters.scene_translation[dim] = spawn.position[dim];
        }
        parameters.scene_yaw_cos = math::cos(device.math, (T)spawn.yaw);
        parameters.scene_yaw_sin = math::sin(device.math, (T)spawn.yaw);
        parameters.shutter_fraction = Config::SHUTTER_FRACTION_MIN;
        initial_state(device, world.dynamics, parameters.agents[0].dynamics, get_ref(device, state.states, (TI)0));
        reset(device, state.pilot, state.rng);
        state.target = {};
        state.moving = false;
        for(TI motor = 0; motor < 4; motor++) {
            state.previous_actions[motor] = 0;
        }
        set_all(device, state.reset_mask, true);
        set_all(device, state.actions, (T)0);
        request_render(device, world, state.reset_mask);
    }
    void control(web::DEVICE& device, web::State& state, web::T forward, web::T left, web::T yaw_delta) {
        using namespace web;
        auto& physical = get_ref(device, state.states, (TI)0);
        const T dt = get_ref(device, state.parameters, (TI)0).agents[0].dynamics.integration.dt;
        update_yaw_target(device, state.pilot, physical, state.target, yaw_delta, dt, state.yaw_rate_limit, state.yaw_error_limit);
        const T norm = math::sqrt(device.math, forward * forward + left * left);
        const bool moving = norm > 0;
        if(moving) {
            const T scale = norm > 1 ? 1 / norm : 1;
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
        observe(device, state.pilot, physical, state.target, state.previous_actions, (TI)0);
        evaluate_step(device, state.pilot, state.rng);
        for(TI motor = 0; motor < 4; motor++) {
            state.previous_actions[motor] = get(device, state.pilot.actions, (TI)0, motor);
        }
        for(TI motor = 0; motor < 4; motor++) {
            set(device, state.actions, state.previous_actions[state.motor_order[motor]], (TI)0, motor);
        }
    }
    void step(web::DEVICE& device, web::State& state, web::T forward, web::T left, web::T yaw_delta, web::TI steps) {
        using namespace web;
        auto& world = state.environment.environments[0];
        auto actions = matrix_view(device, state.actions);
        for(TI step_i = 0; step_i < steps; step_i++) {
            control(device, state, forward, left, yaw_delta / steps);
            // Only the final control tick advances camera/flow history; scalar steps integrate dynamics alone.
            if(step_i + 1 < steps) {
                step(device, world, get_ref(device, state.parameters, (TI)0), get_ref(device, state.states, (TI)0), actions, get_ref(device, state.next_states, (TI)0), state.rng);
            } else {
                step(device, state.environment, state.parameters, state.states, state.actions, state.next_states, state.rng);
            }
            copy(device, device, state.next_states, state.states);
        }
    }
    template <typename TENSOR, typename T>
    void copy_onboard(web::DEVICE& device, web::WORLD::RENDERER& renderer, TENSOR& source, T* destination) {
        auto onboard = view(device, source, (web::TI)0);
        Tensor<typename decltype(onboard)::SPEC> alias;
        alias._data = destination;
        copy(renderer.device, device, onboard, alias);
    }
    void render(web::DEVICE& device, web::State& state, web::Buffer& buffer) {
        using namespace web;
        render(device, state.environment, state.parameters, state.states, state.reset_mask);
        set_all(device, state.reset_mask, false);
        auto& renderer = get_renderer(device, state.environment.environments[0]);
        auto& rgb = frame_buffer(device, renderer);
        Tensor<typename web::WORLD::RENDERER::FB_TENSOR_SPEC> alias;
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
static void read_input(float input[9]) {
    EM_ASM({
        const input = Module.rlt_input();
        for(let i = 0; i < 9; i++) {
            HEAPF32[($0 >> 2) + i] = input[i];
        }
    }, input);
}
static void present(const web::Buffer& buffer) {
    EM_ASM({
        if(Module.rlt_present) {
            Module.rlt_present($0, $1, $2, $3);
        }
    }, buffer.rgba.data(), (int)web::Config::CAM_WIDTH, (int)web::Config::CAM_HEIGHT, (int)web::VIEWS);
}

extern "C" EMSCRIPTEN_KEEPALIVE int demo_run(const char* scene_path, const char* drone_path, const char* policy_path, int drone_index, int max_frames) {
    using namespace rl_tools;
    web::DEVICE device;
    init(device);
    web::State state;
    web::Buffer buffer;
    malloc(device, state);
    status("Loading RAPTOR ...");
    if(!load(device, state.pilot, std::string(policy_path))) {
        status("The RAPTOR checkpoint could not be loaded");
        free(device, state);
        return 1;
    }
    status("Loading the scene and finding a clear starting position ...");
    init(device, state, scene_path, drone_path, drone_index);
    reset(device, state);
    status("Ready — click a camera to fly");
    const double dt = get_ref(device, state.parameters, (web::TI)0).agents[0].dynamics.integration.dt;
    double previous_ms = emscripten_get_now();
    double accumulator = 0;
    bool was_active = false;
    for(int frame = 0; max_frames <= 0 || frame < max_frames;) {
        float input[9] = {};
        read_input(input);
        state.yaw_rate_limit = input[5];
        state.yaw_error_limit = input[6];
        state.position_offset = input[7];
        state.velocity_target = input[8];
        const bool active = max_frames > 0 || input[4] != 0;
        if(input[3] != 0) {
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
            for(web::TI dim = 0; dim < 2; dim++) {
                state.target.position[dim] = physical.position[dim];
                state.target.linear_velocity[dim] = 0;
            }
            state.moving = false;
            accumulator = 0;
        }
        was_active = active;
        if(active) {
            const auto steps = (web::TI)(accumulator / dt);
            step(device, state, input[0], input[1], input[2], steps);
            accumulator -= steps * dt;
        }
        if(frame == 0 || active || input[3] != 0) {
            render(device, state, buffer);
            present(buffer);
            frame++;
        }
        const double elapsed_ms = emscripten_get_now() - now_ms;
        const int sleep_ms = active ? (int)std::max(1.0, web::FRAME_PERIOD * 1000.0 - elapsed_ms) : 16;
        emscripten_sleep(sleep_ms);
    }
    free(device, state);
    return 0;
}

int main() {
    return 0;
}
