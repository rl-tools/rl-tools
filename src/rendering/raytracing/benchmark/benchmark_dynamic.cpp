// Dynamic inside-out benchmark: cameras fly deterministic Catmull-Rom loops through
// collision-validated indoor waypoints, with per-step pose uploads inside the timed loop
// (the RL-rollout interaction pattern). Poses are precomputed into a periodic table so the
// per-step CPU cost is a memcpy and the measurement stays GPU-bound.
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB 0
#define RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD 1

#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_SHADING
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_SHADING 2
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_OUTPUT_MODE
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_OUTPUT_MODE RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGB
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_AA_GRID
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_AA_GRID 1
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_MB_SAMPLES
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_MB_SAMPLES 1
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_CONSUME
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_CONSUME 0
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_TILE
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_TILE 1
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_NUM_CAMERAS
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_NUM_CAMERAS 4096
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_WIDTH
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_WIDTH 64
#endif
#ifndef RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_HEIGHT
#define RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_HEIGHT 64
#endif

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/rendering/raytracing/operations_cpu_mux.h>
#include <rl_tools/rendering/raytracing/scene/procthor/operations_cpu.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using T = float;
using TI = int;

#if RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_SHADING == 0
using SHADING = rlt::rendering::raytracing::Low;
static constexpr const char* SHADING_NAME = "low";
#elif RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_SHADING == 1
using SHADING = rlt::rendering::raytracing::Medium;
static constexpr const char* SHADING_NAME = "medium";
#elif RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_SHADING == 2
using SHADING = rlt::rendering::raytracing::High;
static constexpr const char* SHADING_NAME = "high";
#else
using SHADING = rlt::rendering::raytracing::VeryHigh;
static constexpr const char* SHADING_NAME = "veryhigh";
#endif

#if defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_METAL)
static constexpr const char* BACKEND_NAME = "metal";
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_OPTIX)
static constexpr const char* BACKEND_NAME = "optix";
#elif defined(RL_TOOLS_RENDERING_RAYTRACING_BACKEND_VULKAN)
static constexpr const char* BACKEND_NAME = "vulkan";
#else
static constexpr const char* BACKEND_NAME = "generic";
#endif

static constexpr auto OUTPUT_MODE =
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_OUTPUT_MODE == RL_TOOLS_RENDERING_RAYTRACING_OUTPUT_RGBD
        ? rlt::rendering::raytracing::OutputMode::RGBD
        : rlt::rendering::raytracing::OutputMode::RGB;
static constexpr TI NUM_CAMERAS = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_NUM_CAMERAS;
static constexpr TI NUM_PROBES = 64;
static constexpr TI AA_GRID = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_AA_GRID;
static constexpr TI MB_SAMPLES = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_MB_SAMPLES;
static constexpr bool ENABLE_MB = MB_SAMPLES > 1;
static constexpr bool ENABLE_AA = AA_GRID > 1;
static constexpr bool CONSUME = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_CONSUME != 0;
static constexpr TI TILE = RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_TILE;

using SPEC = rlt::rendering::raytracing::Specification<T, TI,
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_WIDTH,
    RL_TOOLS_RENDERING_RAYTRACING_BENCHMARK_DYNAMIC_HEIGHT,
    NUM_CAMERAS, NUM_PROBES, SHADING, ENABLE_MB, MB_SAMPLES, ENABLE_AA, AA_GRID, OUTPUT_MODE>;
using SCENE_SPEC = rlt::rendering::raytracing::scene::SceneSpecification<T, TI, 512>;
using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using Camera = rlt::rendering::raytracing::Camera<T>;

static constexpr T FOV = SPEC::COS_FOVY;
static constexpr T ASPECT = (T)SPEC::CAM_WIDTH / (T)SPEC::CAM_HEIGHT;
static constexpr TI NUM_WAYPOINTS = 4;
static constexpr TI TRAJECTORY_PERIOD = 600;
static constexpr T YAW_AMPLITUDE = 0.6f;
static constexpr T PITCH_AMPLITUDE = 0.25f;
static constexpr T LOS_MARGIN = 0.1f;
static constexpr T PI = 3.14159265358979323846f;

struct Vec3{
    T v[3];
};

static Vec3 catmull_rom(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, T t){
    Vec3 result;
    const T t2 = t * t;
    const T t3 = t2 * t;
    for(int d = 0; d < 3; d++){
        result.v[d] = 0.5f * (2.0f * p1.v[d]
            + (-p0.v[d] + p2.v[d]) * t
            + (2.0f * p0.v[d] - 5.0f * p1.v[d] + 4.0f * p2.v[d] - p3.v[d]) * t2
            + (-p0.v[d] + 3.0f * p1.v[d] - 3.0f * p2.v[d] + p3.v[d]) * t3);
    }
    return result;
}

static Vec3 catmull_rom_derivative(const Vec3& p0, const Vec3& p1, const Vec3& p2, const Vec3& p3, T t){
    Vec3 result;
    const T t2 = t * t;
    for(int d = 0; d < 3; d++){
        result.v[d] = 0.5f * ((-p0.v[d] + p2.v[d])
            + 2.0f * (2.0f * p0.v[d] - 5.0f * p1.v[d] + 4.0f * p2.v[d] - p3.v[d]) * t
            + 3.0f * (-p0.v[d] + 3.0f * p1.v[d] - 3.0f * p2.v[d] + p3.v[d]) * t2);
    }
    return result;
}

static Camera evaluate_pose(const std::array<Vec3, NUM_WAYPOINTS>& waypoints, T u, T yaw_phase){
    namespace v3 = rlt::rendering::raytracing::vec3;
    u = u - std::floor(u / (T)NUM_WAYPOINTS) * (T)NUM_WAYPOINTS;
    const TI segment = (TI)u;
    const T t = u - (T)segment;
    const Vec3& p0 = waypoints[(segment + NUM_WAYPOINTS - 1) % NUM_WAYPOINTS];
    const Vec3& p1 = waypoints[segment % NUM_WAYPOINTS];
    const Vec3& p2 = waypoints[(segment + 1) % NUM_WAYPOINTS];
    const Vec3& p3 = waypoints[(segment + 2) % NUM_WAYPOINTS];
    const Vec3 position = catmull_rom(p0, p1, p2, p3, t);
    Vec3 velocity = catmull_rom_derivative(p0, p1, p2, p3, t);
    T speed = std::sqrt(velocity.v[0] * velocity.v[0] + velocity.v[1] * velocity.v[1] + velocity.v[2] * velocity.v[2]);
    if(speed < 1e-6f){
        velocity = {1, 0, 0};
        speed = 1;
    }
    const T yaw_offset = YAW_AMPLITUDE * std::sin(2.0f * PI * 2.0f * u / (T)NUM_WAYPOINTS + yaw_phase);
    const T cos_yaw = std::cos(yaw_offset);
    const T sin_yaw = std::sin(yaw_offset);
    T look_direction[3] = {
        (velocity.v[0] * cos_yaw - velocity.v[1] * sin_yaw) / speed,
        (velocity.v[0] * sin_yaw + velocity.v[1] * cos_yaw) / speed,
        velocity.v[2] / speed
    };
    const T pitch_offset = PITCH_AMPLITUDE * std::sin(2.0f * PI * 3.0f * u / (T)NUM_WAYPOINTS + 2.0f * yaw_phase);
    const T look_at[3] = {
        position.v[0] + look_direction[0],
        position.v[1] + look_direction[1],
        position.v[2] + look_direction[2] + pitch_offset
    };
    const T up[3] = {0, 0, 1};
    return rlt::make_camera_data(position.v, look_at, up, FOV, ASPECT);
}

template <typename DEVICE, typename RENDERER>
void upload_poses(DEVICE& device, RENDERER& renderer, const Camera* poses, const Camera* poses_open){
    std::memcpy(rlt::data(renderer.cameras), poses, sizeof(Camera) * NUM_CAMERAS);
    if constexpr(RENDERER::SPEC::ENABLE_MOTION_BLUR){
        std::memcpy(rlt::data(renderer.cameras_open), poses_open, sizeof(Camera) * NUM_CAMERAS);
        rlt::set_motion_blur_cameras(device, renderer, renderer.cameras_open, renderer.cameras);
    }
    else{
        rlt::set_cameras(device, renderer, renderer.cameras);
    }
}

template <typename DEVICE, typename RENDERER>
void consume_outputs(DEVICE& device, RENDERER& renderer){
    rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
    if constexpr(RENDERER::SPEC::HAS_DEPTH){
        rlt::read_depth_buffer(device, renderer, renderer.depth_buffer);
    }
}

template <typename DEVICE, typename RENDERER>
void save_depth_if_available(DEVICE& device, RENDERER& renderer, const char* filename){
    if constexpr(RENDERER::SPEC::HAS_DEPTH){
        rlt::save_depth_image(device, renderer, filename);
    }
    else{
        (void)filename;
    }
}

int main(int ac, char** av){
    std::string model_file;
    std::string dump_dir;
    TI num_steps = 2000;
    TI num_warmup = 20;
    for(int i = 1; i < ac; i++){
        std::string arg = av[i];
        if((arg == "-m" || arg == "--model") && i + 1 < ac){
            model_file = av[++i];
        } else if(arg == "--steps" && i + 1 < ac){
            num_steps = std::atoi(av[++i]);
        } else if(arg == "--dump" && i + 1 < ac){
            dump_dir = av[++i];
        } else if(arg == "-h" || arg == "--help"){
            std::cout << "Usage: " << av[0] << " --model <file> [--steps N] [--dump <dir>]\n";
            return 0;
        } else if(arg[0] != '-'){
            model_file = arg;
        }
    }
    if(model_file.empty()){
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("No model specified. Use --model <file>.");
        return 1;
    }

    DEVICE device;
    rlt::init(device);
    rlt::rendering::raytracing::Renderer<SPEC> renderer;
    rlt::malloc(device, renderer);

    rlt::rendering::raytracing::Scene scene;
    if(!rlt::load<SHADING, SPEC::HAS_RGB>(device, scene, model_file)){
        RL_TOOLS_RENDERING_RAYTRACING_LOG_ERR("Failed to load model: " << model_file);
        return 1;
    }
    if constexpr(TILE > 1){
        float bbox_min[3] = {1e30f, 1e30f, 1e30f}, bbox_max[3] = {-1e30f, -1e30f, -1e30f};
        for(const auto& object : scene.objects){
            for(const auto& mesh : object.meshes){
                for(size_t vertex_i = 0; vertex_i + 2 < mesh.vertices.size(); vertex_i += 3){
                    for(int d = 0; d < 3; d++){
                        bbox_min[d] = std::min(bbox_min[d], mesh.vertices[vertex_i + d]);
                        bbox_max[d] = std::max(bbox_max[d], mesh.vertices[vertex_i + d]);
                    }
                }
            }
        }
        const float tile_offset[2] = {bbox_max[0] - bbox_min[0] + 1.0f, bbox_max[1] - bbox_min[1] + 1.0f};
        const size_t num_original_instances = scene.instances.size();
        for(TI tile_x = 0; tile_x < TILE; tile_x++){
            for(TI tile_y = 0; tile_y < TILE; tile_y++){
                if(tile_x == 0 && tile_y == 0){
                    continue;
                }
                for(size_t instance_i = 0; instance_i < num_original_instances; instance_i++){
                    rlt::rendering::raytracing::Instance instance = scene.instances[instance_i];
                    instance.transform[3] += tile_x * tile_offset[0];
                    instance.transform[7] += tile_y * tile_offset[1];
                    instance.identity = false;
                    scene.instances.push_back(instance);
                }
            }
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Tiled scene " << TILE << "x" << TILE << ": " << scene.instances.size() << " instances");
    }
    rlt::init(device, renderer, scene);
    rlt::generate_probe_directions(device, renderer);

    rlt::rendering::raytracing::scene::procthor::Scene<SCENE_SPEC> scene_procthor;
    rlt::rendering::raytracing::scene::procthor::precompute_indoor_positions(device, scene_procthor, renderer, FOV, ASPECT);
    const TI num_positions = scene_procthor.num_indoor_positions;
    rlt::utils::assert_exit(device, num_positions >= 8, "benchmark_dynamic: too few indoor positions");

    auto position_of = [&](TI pool_index) -> Vec3 {
        const auto& indoor_position = scene_procthor.indoor_positions[pool_index];
        return {indoor_position.position[0], indoor_position.position[1], indoor_position.position[2]};
    };
    auto segment_clear = [&](const std::vector<Vec3>& from, const std::vector<Vec3>& to, std::vector<bool>& clear){
        namespace v3 = rlt::rendering::raytracing::vec3;
        for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
            const T up[3] = {0, 0, 1};
            rlt::set(device, renderer.cameras, rlt::make_camera_data(from[camera_i].v, to[camera_i].v, up, FOV, ASPECT), camera_i);
        }
        rlt::set_cameras(device, renderer, renderer.cameras);
        rlt::render_collision_only(device, renderer);
        const auto* probe_results = rlt::read_collision_results_raw(device, renderer);
        for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
            T delta[3];
            v3::sub(to[camera_i].v, from[camera_i].v, delta);
            const T segment_length = v3::length(delta);
            const auto& forward_probe = probe_results[(size_t)camera_i * NUM_PROBES];
            clear[camera_i] = segment_length > 0.5f && (!forward_probe.hit || forward_probe.distance > segment_length - LOS_MARGIN);
        }
    };

    std::mt19937 rng(0x5EED);
    std::uniform_int_distribution<TI> pool_distribution(0, num_positions - 1);
    std::vector<std::array<TI, NUM_WAYPOINTS>> waypoint_indices(NUM_CAMERAS);
    for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
        waypoint_indices[camera_i][0] = pool_distribution(rng);
    }
    std::vector<Vec3> from(NUM_CAMERAS), to(NUM_CAMERAS), to_next(NUM_CAMERAS);
    std::vector<bool> clear(NUM_CAMERAS), clear_return(NUM_CAMERAS), accepted(NUM_CAMERAS);
    for(TI waypoint_i = 1; waypoint_i < NUM_WAYPOINTS; waypoint_i++){
        const bool closing = waypoint_i == NUM_WAYPOINTS - 1;
        std::fill(accepted.begin(), accepted.end(), false);
        TI num_accepted = 0;
        for(TI attempt = 0; attempt < 32 && num_accepted < NUM_CAMERAS; attempt++){
            for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
                if(!accepted[camera_i]){
                    TI candidate = pool_distribution(rng);
                    while(candidate == waypoint_indices[camera_i][waypoint_i - 1]){
                        candidate = pool_distribution(rng);
                    }
                    waypoint_indices[camera_i][waypoint_i] = candidate;
                }
                from[camera_i] = position_of(waypoint_indices[camera_i][waypoint_i - 1]);
                to[camera_i] = position_of(waypoint_indices[camera_i][waypoint_i]);
                to_next[camera_i] = position_of(waypoint_indices[camera_i][0]);
            }
            segment_clear(from, to, clear);
            if(closing){
                segment_clear(to, to_next, clear_return);
            }
            for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
                if(!accepted[camera_i] && clear[camera_i] && (!closing || clear_return[camera_i])){
                    accepted[camera_i] = true;
                    num_accepted++;
                }
            }
        }
        RL_TOOLS_RENDERING_RAYTRACING_LOG("Waypoint " << waypoint_i << ": " << num_accepted << "/" << NUM_CAMERAS << " LOS-validated");
    }

    RL_TOOLS_RENDERING_RAYTRACING_LOG("Precomputing pose tables (" << TRAJECTORY_PERIOD << " steps x " << NUM_CAMERAS << " cameras)");
    std::vector<Camera> pose_table((size_t)TRAJECTORY_PERIOD * NUM_CAMERAS);
    std::vector<Camera> pose_table_open;
    if constexpr(SPEC::ENABLE_MOTION_BLUR){
        pose_table_open.resize((size_t)TRAJECTORY_PERIOD * NUM_CAMERAS);
    }
    const T du = (T)NUM_WAYPOINTS / (T)TRAJECTORY_PERIOD;
    for(TI camera_i = 0; camera_i < NUM_CAMERAS; camera_i++){
        std::array<Vec3, NUM_WAYPOINTS> waypoints;
        for(TI waypoint_i = 0; waypoint_i < NUM_WAYPOINTS; waypoint_i++){
            waypoints[waypoint_i] = position_of(waypoint_indices[camera_i][waypoint_i]);
        }
        const T u_start = (T)NUM_WAYPOINTS * (T)camera_i / (T)NUM_CAMERAS;
        const T yaw_phase = 2.0f * PI * (T)camera_i / (T)NUM_CAMERAS;
        for(TI step = 0; step < TRAJECTORY_PERIOD; step++){
            const T u = u_start + (T)step * du;
            pose_table[(size_t)step * NUM_CAMERAS + camera_i] = evaluate_pose(waypoints, u, yaw_phase);
            if constexpr(SPEC::ENABLE_MOTION_BLUR){
                pose_table_open[(size_t)step * NUM_CAMERAS + camera_i] = evaluate_pose(waypoints, u - 0.5f * du, yaw_phase);
            }
        }
    }

    auto upload_step = [&](TI step){
        const size_t table_offset = (size_t)(step % TRAJECTORY_PERIOD) * NUM_CAMERAS;
        const Camera* poses_open = pose_table_open.empty() ? nullptr : &pose_table_open[table_offset];
        upload_poses(device, renderer, &pose_table[table_offset], poses_open);
    };

    uint64_t checksum = 0;
    auto accumulate_checksum = [&](){
        constexpr size_t pixel_count = (size_t)SPEC::NUM_CAMERAS * SPEC::CAM_PIXELS;
        const uint32_t* frame = rlt::data(renderer.frame_buffer);
        for(size_t pixel_i = 0; pixel_i < pixel_count; pixel_i += 4097){
            checksum = checksum * 1099511628211ull + frame[pixel_i];
        }
    };

    if(!dump_dir.empty()){
        std::filesystem::create_directories(dump_dir);
    }
    auto dump_step = [&](TI step){
        if(dump_dir.empty()){
            return;
        }
        rlt::render_sync(device, renderer);
        char filename[512];
        std::snprintf(filename, sizeof(filename), "%s/%s_step_%04d.png", dump_dir.c_str(), BACKEND_NAME, step);
        rlt::save_image(device, renderer, filename);
        if constexpr(SPEC::HAS_DEPTH){
            std::snprintf(filename, sizeof(filename), "%s/%s_step_%04d_depth.png", dump_dir.c_str(), BACKEND_NAME, step);
        }
        save_depth_if_available(device, renderer, filename);
    };

    for(TI step = 0; step < num_warmup; step++){
        upload_step(step);
        rlt::render_launch(device, renderer);
    }
    rlt::render_sync(device, renderer);

    RL_TOOLS_RENDERING_RAYTRACING_LOG("Starting dynamic benchmark: " << NUM_CAMERAS << " cameras at "
        << SPEC::CAM_WIDTH << "x" << SPEC::CAM_HEIGHT << " shading=" << SHADING_NAME
        << " aa=" << AA_GRID << " mb=" << MB_SAMPLES << " consume=" << (CONSUME ? 1 : 0)
        << " tile=" << TILE << " steps=" << num_steps);

    std::vector<double> step_times_ms((size_t)num_steps);
    for(TI step = 0; step < num_steps; step++){
        auto step_start = std::chrono::high_resolution_clock::now();
        upload_step(num_warmup + step);
        rlt::render_launch(device, renderer);
        if constexpr(CONSUME){
            rlt::render_sync(device, renderer);
            consume_outputs(device, renderer);
            accumulate_checksum();
        }
        auto step_end = std::chrono::high_resolution_clock::now();
        step_times_ms[step] = std::chrono::duration<double, std::milli>(step_end - step_start).count();
        if(step == 0 || step == num_steps / 2 || step == num_steps - 1){
            dump_step(step);
        }
    }
    rlt::render_sync(device, renderer);
    double wall_ms = 0;
    for(double step_time : step_times_ms){
        wall_ms += step_time;
    }

    if constexpr(!CONSUME){
        rlt::read_frame_buffer(device, renderer, renderer.frame_buffer);
        accumulate_checksum();
    }

    std::vector<double> sorted_times = step_times_ms;
    std::sort(sorted_times.begin(), sorted_times.end());
    double mean_ms = 0;
    for(double step_time : step_times_ms){
        mean_ms += step_time;
    }
    mean_ms /= num_steps;
    const double p50_ms = sorted_times[(size_t)(num_steps * 0.50)];
    const double p99_ms = sorted_times[std::min((size_t)(num_steps * 0.99), (size_t)num_steps - 1)];
    const double max_ms = sorted_times[num_steps - 1];

    const long long total_frames = (long long)num_steps * NUM_CAMERAS;
    long long rays_per_frame = (long long)SPEC::CAM_PIXELS * ((SPEC::HAS_RGB ? SPEC::RGB_SAMPLES : 0) + (SPEC::HAS_DEPTH ? SPEC::DEPTH_SAMPLES : 0));
    const long long total_rays = total_frames * (rays_per_frame + NUM_PROBES);
    const double fps = total_frames / (wall_ms / 1000.0);
    const double mrays = (total_rays / 1e6) / (wall_ms / 1000.0);

    RL_TOOLS_RENDERING_RAYTRACING_LOG("=== DYNAMIC BENCHMARK RESULTS ===");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Backend:            " << BACKEND_NAME);
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Step latency mean:  " << mean_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Step latency p50:   " << p50_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Step latency p99:   " << p99_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Step latency max:   " << max_ms << " ms");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Throughput:         " << fps << " frames/sec");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Camera+probe rays:  " << mrays << " MRays/sec (excludes secondary/shadow rays)");
    RL_TOOLS_RENDERING_RAYTRACING_LOG("  Frame checksum:     " << std::hex << checksum << std::dec);
    std::cout << "CSV,dynamic," << BACKEND_NAME << "," << SHADING_NAME << "," << AA_GRID << "," << MB_SAMPLES << ","
              << (SPEC::HAS_DEPTH ? "rgbd" : "rgb") << "," << (CONSUME ? 1 : 0) << "," << TILE << ","
              << NUM_CAMERAS << "," << SPEC::CAM_WIDTH << "," << SPEC::CAM_HEIGHT << "," << num_steps << ","
              << mean_ms << "," << p50_ms << "," << p99_ms << "," << fps << "," << mrays << ","
              << std::hex << checksum << std::dec << std::endl;

    rlt::free(device, renderer);
    return 0;
}
