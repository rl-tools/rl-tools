#include "harness.h"

#include "../../demo_common.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace vio = hyperdrone_vio;

using DEVICE = vio::DEVICE;
using T = vio::T;
using TI = vio::TI;

// exports one benchmark episode in the ASL/EuRoC dataset layout so any visual-inertial
// estimator (OpenVINS, VINS-Fusion, ORB-SLAM3, ...) can consume it; ground truth is the
// protocol's relative-to-episode-start pose
using WORLD = vio::BaselineWorld;
using BASE_WORLD = vio::BaselineBaseWorld;
constexpr TI CAM_WIDTH = WORLD::SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD::SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI DEFAULT_STEPS = 4000; // 20 s at 200 Hz (2.5 s initialization hold + flight)

static long long to_nanoseconds(double time){
    return (long long)(time * 1e9 + 0.5);
}

struct ExportSink {
    DEVICE device;
    std::filesystem::path root;
    std::ofstream camera_csv, imu_csv, ground_truth_csv;
    std::vector<std::uint8_t> grayscale;
    TI frames_written = 0;

    void begin(const vio::EpisodeInfo<WORLD>& info){
        rlt::init(device);
        std::filesystem::create_directories(root / "mav0" / "cam0" / "data");
        std::filesystem::create_directories(root / "mav0" / "imu0");
        std::filesystem::create_directories(root / "mav0" / "state_groundtruth_estimate0");
        camera_csv.open(root / "mav0" / "cam0" / "data.csv");
        camera_csv << "#timestamp [ns],filename\n";
        imu_csv.open(root / "mav0" / "imu0" / "data.csv");
        imu_csv << "#timestamp [ns],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n";
        ground_truth_csv.open(root / "mav0" / "state_groundtruth_estimate0" / "data.csv");
        ground_truth_csv << "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]\n";
        grayscale.resize(CAM_PIXELS);
        {
            std::ofstream yaml(root / "mav0" / "cam0" / "sensor.yaml");
            yaml << "sensor_type: camera\ncomment: rl-tools hyperdrone visual_inertial_localization (ideal pinhole, zero distortion, camera clock == IMU clock)\n";
            yaml << "T_BS:\n  cols: 4\n  rows: 4\n  data: [";
            for(TI row = 0; row < 3; row++){
                for(TI col = 0; col < 3; col++){
                    yaml << info.camera_rotation_body[row][col] << ", ";
                }
                yaml << info.camera_translation_body[row] << ",\n         ";
            }
            yaml << "0.0, 0.0, 0.0, 1.0]\n";
            yaml << "rate_hz: " << info.frame_rate << "\n";
            yaml << "resolution: [" << CAM_WIDTH << ", " << CAM_HEIGHT << "]\n";
            yaml << "camera_model: pinhole\n";
            yaml << "intrinsics: [" << info.intrinsics.fx << ", " << info.intrinsics.fy << ", " << info.intrinsics.cx << ", " << info.intrinsics.cy << "]\n";
            yaml << "distortion_model: radial-tangential\n";
            yaml << "distortion_coefficients: [0.0, 0.0, 0.0, 0.0]\n";
        }
        {
            std::ofstream yaml(root / "mav0" / "imu0" / "sensor.yaml");
            yaml << "sensor_type: imu\ncomment: rl-tools hyperdrone visual_inertial_localization (continuous-time densities, Kalibr convention)\n";
            yaml << "T_BS:\n  cols: 4\n  rows: 4\n  data: [1.0, 0.0, 0.0, 0.0,\n         0.0, 1.0, 0.0, 0.0,\n         0.0, 0.0, 1.0, 0.0,\n         0.0, 0.0, 0.0, 1.0]\n";
            yaml << "rate_hz: " << info.imu_rate << "\n";
            yaml << "gyroscope_noise_density: " << info.gyro_noise_density << "\n";
            yaml << "gyroscope_random_walk: " << info.gyro_random_walk << "\n";
            yaml << "accelerometer_noise_density: " << info.accelerometer_noise_density << "\n";
            yaml << "accelerometer_random_walk: " << info.accelerometer_random_walk << "\n";
        }
    }
    template <typename OBSERVATIONS>
    void frame(double time, TI, const OBSERVATIONS& observations){
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            T gray = (T)0.299 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 0)
                   + (T)0.587 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 1)
                   + (T)0.114 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 2);
            gray = gray < 0 ? 0 : (gray > 1 ? (T)1 : gray);
            grayscale[pixel_i] = (std::uint8_t)(gray * (T)255 + (T)0.5);
        }
        long long t_ns = to_nanoseconds(time);
        std::string filename = std::to_string(t_ns) + ".png";
        stbi_write_png((root / "mav0" / "cam0" / "data" / filename).c_str(), (int)CAM_WIDTH, (int)CAM_HEIGHT, 1, grayscale.data(), (int)CAM_WIDTH);
        camera_csv << t_ns << "," << filename << "\n";
        frames_written++;
    }
    template <typename OBSERVATIONS_IMU>
    void step(double time, const OBSERVATIONS_IMU& observations_imu, const T relative_positions[][3], const T relative_orientations[][4], const T relative_velocities[][3], vio::Tensors<WORLD>& tensors){
        long long t_ns = to_nanoseconds(time);
        imu_csv << t_ns;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            imu_csv << "," << rlt::get(device, observations_imu, (TI)0, 3 + dim_i); // gyro first (EuRoC column order)
        }
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            imu_csv << "," << rlt::get(device, observations_imu, (TI)0, dim_i);
        }
        imu_csv << "\n";
        typename WORLD::State next_state = rlt::get(device, tensors.next_states, (TI)0);
        ground_truth_csv << t_ns;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            ground_truth_csv << "," << relative_positions[0][dim_i];
        }
        for(TI dim_i = 0; dim_i < 4; dim_i++){
            ground_truth_csv << "," << relative_orientations[0][dim_i]; // w, x, y, z (Hamilton)
        }
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            ground_truth_csv << "," << relative_velocities[0][dim_i];
        }
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            ground_truth_csv << "," << next_state.gyro_bias[dim_i];
        }
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            ground_truth_csv << "," << next_state.accelerometer_bias[dim_i];
        }
        ground_truth_csv << "\n";
    }
};

int main(int argc, char** argv){
#ifdef RL_TOOLS_TEST_DATA_PATH
    std::string scene_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
    std::string scene_path = "";
#endif
    std::string output_dir = "visual_inertial_localization_euroc";
    TI seed = 0;
    TI steps = DEFAULT_STEPS;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        output_dir = argv[2];
    }
    if(argc > 3){
        seed = std::stoul(argv[3]);
    }
    if(argc > 4){
        steps = std::stoul(argv[4]);
    }
    if(scene_path.empty()){
        std::fprintf(stderr, "usage: %s <scene.glb> [output_dir] [seed] [steps]\n", argv[0]);
        return 1;
    }
    ExportSink sink;
    sink.root = output_dir;
    auto result = vio::run_episode<WORLD, BASE_WORLD>(scene_path, seed, steps, sink);
    std::printf("exported %s (%lu frames at %lux%lu, %lu IMU samples, %.1f s wall clock)%s\n",
        output_dir.c_str(), (unsigned long)sink.frames_written, (unsigned long)CAM_WIDTH, (unsigned long)CAM_HEIGHT,
        (unsigned long)steps, result.wall_clock_time, result.any_terminated ? " [warning: terminated]" : "");
    return 0;
}
