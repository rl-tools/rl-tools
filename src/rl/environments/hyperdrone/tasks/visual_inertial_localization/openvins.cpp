#include "harness.h"
#include <rl_tools/rl/environments/hyperdrone/tasks/visual_inertial_localization/baseline.h>

#include "../../demo_common.h"

#include <metra/metra.h>

// OpenVINS (GPL-3.0, built out-of-tree by src/vio/openvins/setup.sh — this binary is GPL,
// for internal benchmarking only)
#include <core/VioManager.h>
#include <core/VioManagerOptions.h>
#include <state/State.h>
#include <cam/CamRadtan.h>
#include <utils/sensor_data.h>
#include <utils/quat_ops.h>
#include <utils/print.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace vio = hyperdrone_vio;
namespace task = vio::task;

using DEVICE = vio::DEVICE;
using T = vio::T;
using TI = vio::TI;

// causal OpenVINS (MSCKF) baseline on the benchmark's baseline configuration, with the
// dead-reckoning floor computed side by side for an apples-to-apples comparison
using WORLD = vio::BaselineWorld;
using BASE_WORLD = vio::BaselineBaseWorld;
constexpr TI CAM_WIDTH = WORLD::SPEC::CAM_WIDTH;
constexpr TI CAM_HEIGHT = WORLD::SPEC::CAM_HEIGHT;
constexpr TI CAM_PIXELS = CAM_WIDTH * CAM_HEIGHT;
constexpr TI DEFAULT_STEPS = 4000; // 20 s at 200 Hz (2.5 s initialization hold + flight)
constexpr double INIT_IMU_THRESHOLD = 0.9; // accel std [m/s^2] separating hover (~0.5, dominated by the 0.28 white noise) from the launch jerk (~1.4)

struct OpenVinsSink {
    DEVICE device;
    T dt;
    T gravity[3];
    std::shared_ptr<ov_msckf::VioManager> manager;
    bool has_pending_frame = false;
    double pending_frame_time = 0;
    cv::Mat pending_frame;
    cv::Mat mask;
    // anchored at the estimator's first valid pose; scored against the ground truth anchored at
    // the same timestamp (causal: no ground truth reaches the estimator)
    bool anchored = false;
    bool diverged = false;
    double anchor_time = 0;
    T estimate_anchor_position[3];
    T estimate_anchor_orientation[4];
    T ground_truth_anchor_position[3];
    T ground_truth_anchor_orientation[4];
    T last_estimate_position[3];
    T last_estimate_orientation[4];
    bool has_estimate = false;
    task::TrajectoryMetricsAccumulator<T, TI> openvins_metrics;
    task::DeadReckoningState<T> dead_reckoning;
    task::TrajectoryMetricsAccumulator<T, TI> dead_reckoning_metrics;
    std::ofstream trajectory_tum;

    void begin(const vio::EpisodeInfo<WORLD>& info){
        rlt::init(device);
        dt = info.dt;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            gravity[dim_i] = info.gravity[dim_i];
            dead_reckoning.position[dim_i] = 0;
            dead_reckoning.linear_velocity[dim_i] = 0;
        }
        dead_reckoning.orientation[0] = 1;
        for(TI dim_i = 1; dim_i < 4; dim_i++){
            dead_reckoning.orientation[dim_i] = 0;
        }
        task::reset(device, openvins_metrics);
        task::reset(device, dead_reckoning_metrics);
        mask = cv::Mat::zeros((int)CAM_HEIGHT, (int)CAM_WIDTH, CV_8UC1);

        ov_msckf::VioManagerOptions params;
        params.state_options.num_cameras = 1;
        params.state_options.do_calib_camera_pose = false;
        params.state_options.do_calib_camera_intrinsics = false;
        params.state_options.do_calib_camera_timeoffset = false;
        params.imu_noises.sigma_w = info.gyro_noise_density;
        params.imu_noises.sigma_wb = info.gyro_random_walk;
        params.imu_noises.sigma_a = info.accelerometer_noise_density;
        params.imu_noises.sigma_ab = info.accelerometer_random_walk;
        params.gravity_mag = -(double)info.gravity[2];
        // the IMU intrinsics have no default initializers (the YAML loader computes them);
        // programmatic construction must set the identity model explicitly, otherwise Da/Dw are
        // zero and every measurement collapses to a_hat = w_hat = 0
        params.vec_dw << 1, 0, 0, 1, 0, 1;
        params.vec_da << 1, 0, 0, 1, 0, 1;
        params.vec_tg.setZero();
        params.q_ACCtoIMU << 0, 0, 0, 1;
        params.q_GYROtoIMU << 0, 0, 0, 1;
        params.calib_camimu_dt = 0.0;
        params.use_klt = true;
        params.use_aruco = false; // ENABLE_ARUCO_TAGS=OFF in the OpenVINS build
        params.state_options.max_aruco_features = 0;
        params.num_opencv_threads = 1;
        params.use_multi_threading_subs = false;
        // a hovering drone is never optically still (attitude jitter exceeds the 1px disparity
        // gate) and the hold may face low-texture walls, so initialize from the IMU alone via
        // the ZUPT path: zero-velocity updates pin the state through the stationary hold and
        // stop for good once the launch happens
        params.try_zupt = true;
        params.zupt_only_at_beginning = true;
        params.zupt_max_velocity = 0.5;
        params.zupt_max_disparity = 0.5; // hover inter-frame disparity is ~0.02 px; flight is far above
        auto camera = std::make_shared<ov_core::CamRadtan>((int)CAM_WIDTH, (int)CAM_HEIGHT);
        Eigen::MatrixXd camera_values(8, 1);
        camera_values << info.intrinsics.fx, info.intrinsics.fy, info.intrinsics.cx, info.intrinsics.cy, 0, 0, 0, 0;
        camera->set_value(camera_values);
        params.camera_intrinsics[0] = camera;
        Eigen::Matrix3d rotation_camera_to_imu;
        Eigen::Vector3d translation_camera_in_imu;
        for(TI row = 0; row < 3; row++){
            for(TI col = 0; col < 3; col++){
                rotation_camera_to_imu(row, col) = info.camera_rotation_body[row][col];
            }
            translation_camera_in_imu(row) = info.camera_translation_body[row];
        }
        Eigen::VectorXd extrinsics(7);
        extrinsics.block(0, 0, 4, 1) = ov_core::rot_2_quat(rotation_camera_to_imu.transpose());
        extrinsics.block(4, 0, 3, 1) = -rotation_camera_to_imu.transpose() * translation_camera_in_imu;
        params.camera_extrinsics[0] = extrinsics;
        // the initializer options mirror the sensor configuration and are not synced automatically
        params.init_options.num_cameras = 1;
        params.init_options.sigma_w = params.imu_noises.sigma_w;
        params.init_options.sigma_wb = params.imu_noises.sigma_wb;
        params.init_options.sigma_a = params.imu_noises.sigma_a;
        params.init_options.sigma_ab = params.imu_noises.sigma_ab;
        params.init_options.gravity_mag = params.gravity_mag;
        params.init_options.camera_intrinsics = params.camera_intrinsics;
        params.init_options.camera_extrinsics = params.camera_extrinsics;
        params.init_options.init_window_time = 1.0;
        params.init_options.init_imu_thresh = INIT_IMU_THRESHOLD;
        params.init_options.init_max_disparity = 20.0; // hover attitude jitter is a few px at fx~257; flight is 70+ px
        params.init_options.init_dyn_use = false;
        if(std::getenv("OPENVINS_DEBUG") != nullptr){
            ov_core::Printer::setPrintLevel(ov_core::Printer::PrintLevel::DEBUG);
        }
        manager = std::make_shared<ov_msckf::VioManager>(params);
    }
    template <typename OBSERVATIONS>
    void frame(double time, TI, const OBSERVATIONS& observations){
        cv::Mat gray((int)CAM_HEIGHT, (int)CAM_WIDTH, CV_8UC1);
        for(TI pixel_i = 0; pixel_i < CAM_PIXELS; pixel_i++){
            T value = (T)0.299 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 0)
                    + (T)0.587 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 1)
                    + (T)0.114 * rlt::get(device, observations, (TI)0, pixel_i * 3 + 2);
            value = value < 0 ? 0 : (value > 1 ? (T)1 : value);
            gray.data[pixel_i] = (unsigned char)(value * (T)255 + (T)0.5);
        }
        pending_frame = gray;
        pending_frame_time = time;
        has_pending_frame = true;
    }
    template <typename OBSERVATIONS_IMU>
    void step(double time, const OBSERVATIONS_IMU& observations_imu, const T relative_positions[][3], const T relative_orientations[][4], const T[][3], vio::Tensors<WORLD>&){
        T accelerometer[3], gyroscope[3];
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            accelerometer[dim_i] = rlt::get(device, observations_imu, (TI)0, dim_i);
            gyroscope[dim_i] = rlt::get(device, observations_imu, (TI)0, 3 + dim_i);
        }
        task::dead_reckoning_step(device, dead_reckoning, accelerometer, gyroscope, gravity, dt);
        task::accumulate(device, dead_reckoning_metrics, relative_positions[0], relative_orientations[0], dead_reckoning.position, dead_reckoning.orientation);
        if(diverged){
            return;
        }
        ov_core::ImuData imu;
        imu.timestamp = time;
        for(TI dim_i = 0; dim_i < 3; dim_i++){
            imu.wm(dim_i) = gyroscope[dim_i];
            imu.am(dim_i) = accelerometer[dim_i];
        }
        manager->feed_measurement_imu(imu);
        // camera frames wait until an IMU sample newer than them exists (serial feeding pattern)
        if(has_pending_frame && pending_frame_time <= time){
            ov_core::CameraData camera;
            camera.timestamp = pending_frame_time;
            camera.sensor_ids.push_back(0);
            camera.images.push_back(pending_frame);
            camera.masks.push_back(mask);
            manager->feed_measurement_camera(camera);
            has_pending_frame = false;
            if(manager->initialized()){
                auto state = manager->get_state();
                Eigen::Matrix<double, 4, 1> quat_jpl = state->_imu->quat(); // JPL q_GtoI (x,y,z,w) == Hamilton q_ItoG
                Eigen::Vector3d position = state->_imu->pos();
                T estimate_position[3], estimate_orientation[4];
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    estimate_position[dim_i] = (T)position(dim_i);
                }
                estimate_orientation[0] = (T)quat_jpl(3);
                estimate_orientation[1] = (T)quat_jpl(0);
                estimate_orientation[2] = (T)quat_jpl(1);
                estimate_orientation[3] = (T)quat_jpl(2);
                bool finite = true;
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    finite = finite && std::isfinite(estimate_position[dim_i]);
                }
                for(TI dim_i = 0; dim_i < 4; dim_i++){
                    finite = finite && std::isfinite(estimate_orientation[dim_i]);
                }
                if(!finite){
                    diverged = true;
                    return;
                }
                if(!anchored){
                    anchored = true;
                    anchor_time = pending_frame_time;
                    for(TI dim_i = 0; dim_i < 3; dim_i++){
                        estimate_anchor_position[dim_i] = estimate_position[dim_i];
                        ground_truth_anchor_position[dim_i] = relative_positions[0][dim_i];
                    }
                    for(TI dim_i = 0; dim_i < 4; dim_i++){
                        estimate_anchor_orientation[dim_i] = estimate_orientation[dim_i];
                        ground_truth_anchor_orientation[dim_i] = relative_orientations[0][dim_i];
                    }
                }
                T estimate_relative_position[3], estimate_relative_orientation[4];
                T ground_truth_relative_position[3], ground_truth_relative_orientation[4];
                task::relative_pose(device, estimate_anchor_position, estimate_anchor_orientation, estimate_position, estimate_orientation, estimate_relative_position, estimate_relative_orientation);
                task::relative_pose(device, ground_truth_anchor_position, ground_truth_anchor_orientation, relative_positions[0], relative_orientations[0], ground_truth_relative_position, ground_truth_relative_orientation);
                task::accumulate(device, openvins_metrics, ground_truth_relative_position, ground_truth_relative_orientation, estimate_relative_position, estimate_relative_orientation);
                for(TI dim_i = 0; dim_i < 3; dim_i++){
                    last_estimate_position[dim_i] = estimate_relative_position[dim_i];
                }
                for(TI dim_i = 0; dim_i < 4; dim_i++){
                    last_estimate_orientation[dim_i] = estimate_relative_orientation[dim_i];
                }
                has_estimate = true;
                if(trajectory_tum.is_open()){
                    trajectory_tum << pending_frame_time << " "
                        << estimate_relative_position[0] << " " << estimate_relative_position[1] << " " << estimate_relative_position[2] << " "
                        << estimate_relative_orientation[1] << " " << estimate_relative_orientation[2] << " " << estimate_relative_orientation[3] << " " << estimate_relative_orientation[0] << "\n";
                }
            }
        }
    }
};

int main(int argc, char** argv){
#ifdef RL_TOOLS_TEST_DATA_PATH
    std::string scene_path = std::string(HYPERDRONE_DEMO_STRINGIFY(RL_TOOLS_TEST_DATA_PATH)) + "/ProcTHOR-Train-1.glb";
#else
    std::string scene_path = "";
#endif
    std::string trajectory_path = "visual_inertial_localization_openvins.tum";
    TI seed = 0;
    TI steps = DEFAULT_STEPS;
    if(argc > 1){
        scene_path = argv[1];
    }
    if(argc > 2){
        trajectory_path = argv[2];
    }
    if(argc > 3){
        seed = std::stoul(argv[3]);
    }
    if(argc > 4){
        steps = std::stoul(argv[4]);
    }
    if(scene_path.empty()){
        std::fprintf(stderr, "usage: %s <scene.glb> [trajectory.tum] [seed] [steps]\n", argv[0]);
        return 1;
    }
    OpenVinsSink sink;
    sink.trajectory_tum.open(trajectory_path);
    auto result = vio::run_episode<WORLD, BASE_WORLD>(scene_path, seed, steps, sink);

    if(sink.diverged){
        std::printf("openvins: DIVERGED (non-finite state)\n");
    } else if(!sink.anchored){
        std::printf("openvins: never initialized (%lu steps)\n", (unsigned long)steps);
    } else {
        std::printf("openvins (init at t=%.3f s):\n", sink.anchor_time);
        std::printf("  ATE %.3f m | rot mean %.4f rad max %.4f rad | drift %.4f m/m | distance %.2f m\n",
            (double)task::ate_position_rmse(sink.device, sink.openvins_metrics),
            (double)task::mean_rotation_error(sink.device, sink.openvins_metrics),
            (double)task::max_rotation_error(sink.device, sink.openvins_metrics),
            (double)task::drift_per_distance(sink.device, sink.openvins_metrics),
            (double)sink.openvins_metrics.distance_traveled);
    }
    std::printf("dead reckoning (same episode):\n");
    std::printf("  ATE %.3f m | rot mean %.4f rad max %.4f rad | drift %.4f m/m | distance %.2f m\n",
        (double)task::ate_position_rmse(sink.device, sink.dead_reckoning_metrics),
        (double)task::mean_rotation_error(sink.device, sink.dead_reckoning_metrics),
        (double)task::max_rotation_error(sink.device, sink.dead_reckoning_metrics),
        (double)task::drift_per_distance(sink.device, sink.dead_reckoning_metrics),
        (double)sink.dead_reckoning_metrics.distance_traveled);
    std::printf("wall clock: %.1f s%s\n", result.wall_clock_time, result.any_terminated ? " [warning: terminated]" : "");

    const std::string metra_prefix = "hyperdrone/visual_inertial_localization/openvins";
    metra::log(metra_prefix + "/diverged", sink.diverged ? 1.0 : 0.0);
    metra::log(metra_prefix + "/initialized", sink.anchored ? 1.0 : 0.0);
    if(sink.anchored && !sink.diverged){
        metra::log(metra_prefix + "/initialization_time", sink.anchor_time);
        metra::log(metra_prefix + "/ate_position_rmse", (double)task::ate_position_rmse(sink.device, sink.openvins_metrics));
        metra::log(metra_prefix + "/rotation_error_mean", (double)task::mean_rotation_error(sink.device, sink.openvins_metrics));
        metra::log(metra_prefix + "/rotation_error_max", (double)task::max_rotation_error(sink.device, sink.openvins_metrics));
        metra::log(metra_prefix + "/drift_per_distance", (double)task::drift_per_distance(sink.device, sink.openvins_metrics));
        metra::log(metra_prefix + "/distance_traveled", (double)sink.openvins_metrics.distance_traveled);
    }
    metra::log(metra_prefix + "/dead_reckoning_reference/ate_position_rmse", (double)task::ate_position_rmse(sink.device, sink.dead_reckoning_metrics));
    metra::log(metra_prefix + "/wall_clock_time", result.wall_clock_time);
    return sink.diverged || !sink.anchored ? 1 : 0;
}
