#include "visual_inertial_localization_harness.h"
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/baseline.h>

#include "../demo_common.h"

#include <metra/metra.h>

#include <cstdio>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace vio = hyperdrone_vio;
namespace task = vio::task;

using DEVICE = vio::DEVICE;
using T = vio::T;
using TI = vio::TI;

using WORLD = vio::DefaultWorld;
using BASE_WORLD = vio::DefaultBaseWorld;
constexpr TI INSTANCES = WORLD::INSTANCES;
constexpr TI CAM_WIDTH = WORLD::SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD::SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI STEPS = vio::DefaultWorldSpec::DYNAMICS_STATIC_PARAMETERS::EPISODE_STEP_LIMIT;
constexpr TI UPSCALE = 8;

static_assert(BASE_WORLD::IMAGE_CHANNELS == 3);

// dead-reckoning baseline (blind: identity/zero init per protocol) + oracle metric sanity,
// rendered to video, all fed causally from the shared episode harness
struct DemoSink {
    DEVICE device;
    std::string output_path;
    FILE* video_pipe = nullptr;
    std::vector<std::uint8_t> frame_buffer;
    T dt;
    T gravity[3];
    task::DeadReckoningState<T> estimators[INSTANCES];
    task::TrajectoryMetricsAccumulator<T, TI> metrics[INSTANCES];
    task::TrajectoryMetricsAccumulator<T, TI> oracle_metrics[INSTANCES];
    TI frames_written = 0;
    bool video_ok = true;

    void begin(const vio::EpisodeInfo<WORLD>& info){
        rlt::init(device);
        dt = info.dt;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            gravity[dim_i] = info.gravity[dim_i];
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                estimators[instance_i].position[dim_i] = 0;
                estimators[instance_i].linear_velocity[dim_i] = 0;
            }
            estimators[instance_i].orientation[0] = 1;
            for(TI dim_i = 1; dim_i < 4; dim_i++){
                estimators[instance_i].orientation[dim_i] = 0;
            }
            task::reset(device, metrics[instance_i]);
            task::reset(device, oracle_metrics[instance_i]);
        }
        video_pipe = hyperdrone_demo::open_video_pipe(CAM_WIDTH, CAM_HEIGHT * INSTANCES, (std::size_t)(info.frame_rate + (T)0.5), UPSCALE, output_path);
        frame_buffer.resize(INSTANCES * CAM_PIXELS * 3);
    }
    template <typename OBSERVATIONS>
    void frame(double, TI, const OBSERVATIONS& observations){
        if(video_pipe == nullptr){
            return;
        }
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
                for(TI channel_i = 0; channel_i < 3; channel_i++){
                    frame_buffer[(instance_i * CAM_PIXELS + pixel_i) * 3 + channel_i] = hyperdrone_demo::quantize_pixel(rlt::get(device, observations, instance_i, pixel_i * 3 + channel_i));
                }
            }
        }
        std::fwrite(frame_buffer.data(), 1, frame_buffer.size(), video_pipe);
        frames_written++;
    }
    template <typename OBSERVATIONS_IMU>
    void step(double, const OBSERVATIONS_IMU& observations_imu, const T relative_positions[][3], const T relative_orientations[][4], const T[][3], vio::Tensors<WORLD>&){
        for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
            T accelerometer[3], gyroscope[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                accelerometer[dim_i] = rlt::get(device, observations_imu, instance_i, dim_i);
                gyroscope[dim_i] = rlt::get(device, observations_imu, instance_i, 3 + dim_i);
            }
            task::dead_reckoning_step(device, estimators[instance_i], accelerometer, gyroscope, gravity, dt);
            task::accumulate(device, metrics[instance_i], relative_positions[instance_i], relative_orientations[instance_i], estimators[instance_i].position, estimators[instance_i].orientation);
            task::accumulate(device, oracle_metrics[instance_i], relative_positions[instance_i], relative_orientations[instance_i], relative_positions[instance_i], relative_orientations[instance_i]);
        }
    }
    bool finish(){
        video_ok = video_pipe != nullptr && hyperdrone_demo::close_video_pipe(video_pipe);
        video_pipe = nullptr;
        return video_ok;
    }
};

int main(int argc, char** argv){
#ifdef RL_TOOLS_TEST_DATA_PATH
    std::string scene_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
    std::string scene_path = "";
#endif
    std::string output_path = "visual_inertial_localization.mp4";
    TI seed = 0;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        output_path = argv[2];
    }
    if(argc > 3){
        seed = std::stoul(argv[3]);
    }
    if(scene_path.empty()){
        std::fprintf(stderr, "usage: %s <scene.glb> [output.mp4] [seed]\n", argv[0]);
        return 1;
    }

    DemoSink sink;
    sink.output_path = output_path;
    auto result = vio::run_episode<WORLD, BASE_WORLD>(scene_path, seed, STEPS, sink);
    bool video_ok = sink.finish();
    if(!video_ok){
        std::fprintf(stderr, "ffmpeg exited with an error for %s\n", output_path.c_str());
    } else {
        std::printf("wrote %s (%lu frames, %lu IMU samples)\n", output_path.c_str(), (unsigned long)sink.frames_written, (unsigned long)STEPS);
    }
    std::vector<double> ate_position_rmse_values, rotation_error_mean_values, rotation_error_max_values, drift_per_distance_values, distance_traveled_values, oracle_ate_values;
    for(TI instance_i = 0; instance_i < INSTANCES; instance_i++){
        std::printf("instance %lu:\n", (unsigned long)instance_i);
        std::printf("  oracle          ATE %.6f m (must be 0)\n", (double)task::ate_position_rmse(sink.device, sink.oracle_metrics[instance_i]));
        std::printf("  dead reckoning  ATE %.3f m | rot mean %.4f rad max %.4f rad | drift %.4f m/m | distance %.2f m\n",
            (double)task::ate_position_rmse(sink.device, sink.metrics[instance_i]),
            (double)task::mean_rotation_error(sink.device, sink.metrics[instance_i]),
            (double)task::max_rotation_error(sink.device, sink.metrics[instance_i]),
            (double)task::drift_per_distance(sink.device, sink.metrics[instance_i]),
            (double)sink.metrics[instance_i].distance_traveled);
        ate_position_rmse_values.push_back((double)task::ate_position_rmse(sink.device, sink.metrics[instance_i]));
        rotation_error_mean_values.push_back((double)task::mean_rotation_error(sink.device, sink.metrics[instance_i]));
        rotation_error_max_values.push_back((double)task::max_rotation_error(sink.device, sink.metrics[instance_i]));
        drift_per_distance_values.push_back((double)task::drift_per_distance(sink.device, sink.metrics[instance_i]));
        distance_traveled_values.push_back((double)sink.metrics[instance_i].distance_traveled);
        oracle_ate_values.push_back((double)task::ate_position_rmse(sink.device, sink.oracle_metrics[instance_i]));
    }
    if(result.any_terminated){
        std::printf("warning: at least one instance terminated during the episode\n");
    }
    const std::string metra_prefix = "hyperdrone/visual_inertial_localization";
    metra::log(metra_prefix + "/dead_reckoning/ate_position_rmse", ate_position_rmse_values);
    metra::log(metra_prefix + "/dead_reckoning/rotation_error_mean", rotation_error_mean_values);
    metra::log(metra_prefix + "/dead_reckoning/rotation_error_max", rotation_error_max_values);
    metra::log(metra_prefix + "/dead_reckoning/drift_per_distance", drift_per_distance_values);
    metra::log(metra_prefix + "/distance_traveled", distance_traveled_values);
    metra::log(metra_prefix + "/oracle/ate_position_rmse", oracle_ate_values);
    metra::log(metra_prefix + "/episode/wall_clock_time", result.wall_clock_time);
    metra::log(metra_prefix + "/episode/steps_per_second", (double)(STEPS * INSTANCES) / result.wall_clock_time);
    return video_ok ? 0 : 1;
}
