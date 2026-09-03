#pragma once

#include "harness.h"

#include "../../demo_common.h"

#include <core/VioManager.h>
#include <state/State.h>
#include <cam/CamBase.h>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// annotated video of an OpenVINS run: [feature tracks | landmarks projected into the frame |
// top-down trace], drawn at SCALE x the camera resolution so markers and text stay legible
namespace hyperdrone_vio {
    // rigid transform from the estimator's global frame into the benchmark's episode-start frame,
    // fixed by the same anchor pair the metrics use
    struct FrameAlignment {
        T estimate_anchor_position[3];
        T estimate_anchor_orientation[4];
        T ground_truth_anchor_position[3];
        T ground_truth_anchor_orientation[4];
        void rotate(const T direction[3], T out[3]) const {
            T conjugate[4] = {estimate_anchor_orientation[0], -estimate_anchor_orientation[1], -estimate_anchor_orientation[2], -estimate_anchor_orientation[3]};
            T relative[3];
            rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(conjugate, direction, relative);
            rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(ground_truth_anchor_orientation, relative, out);
        }
        void apply(const T position[3], T out[3]) const {
            T delta[3];
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                delta[dim_i] = position[dim_i] - estimate_anchor_position[dim_i];
            }
            rotate(delta, out);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                out[dim_i] += ground_truth_anchor_position[dim_i];
            }
        }
    };

    inline const cv::Scalar COLOR_GROUND_TRUTH(255, 255, 255);
    inline const cv::Scalar COLOR_ESTIMATE(0, 165, 255);
    inline const cv::Scalar COLOR_SLAM(0, 255, 0);
    inline const cv::Scalar COLOR_MSCKF(255, 0, 255);
    inline const cv::Scalar COLOR_MAP(140, 140, 140);
    inline const cv::Scalar COLOR_GRID(48, 48, 48);
    inline const cv::Scalar COLOR_BACKGROUND(24, 24, 24);
    inline const cv::Scalar COLOR_TEXT(255, 255, 255);
    inline const cv::Scalar COLOR_TEXT_OUTLINE(0, 0, 0);
    inline const cv::Scalar COLOR_WARNING(0, 0, 255);

    template <TI T_CAM_WIDTH, TI T_CAM_HEIGHT>
    struct OpenVinsVideo {
        static constexpr TI CAM_WIDTH = T_CAM_WIDTH;
        static constexpr TI CAM_HEIGHT = T_CAM_HEIGHT;
        static constexpr TI SCALE = 2;
        static constexpr TI PANEL_WIDTH = CAM_WIDTH * SCALE;
        static constexpr TI PANEL_HEIGHT = CAM_HEIGHT * SCALE;
        static constexpr TI TRACE_PANEL_WIDTH = PANEL_HEIGHT;
        static constexpr TI WIDTH = 2 * PANEL_WIDTH + TRACE_PANEL_WIDTH;
        static constexpr TI HEIGHT = PANEL_HEIGHT;
        static constexpr TI MAX_MAP_POINTS = 20000;
        static constexpr TI TRACE_BORDER = 24;
        static constexpr double DEPTH_NEAR = 0.5;
        static constexpr double DEPTH_FAR = 8.0;
        static constexpr T TRACE_MARGIN = 1.0;
        static constexpr T TRACE_MIN_EXTENT = 2.0;

        struct Landmark {
            std::size_t id;
            Eigen::Vector3d position; // estimator global frame
            Eigen::Vector3d measurement; // tracked pixel u, v and estimated depth in the current frame
            bool slam;
        };
        struct Frame {
            double time;
            const cv::Mat* image; // grayscale camera frame as fed to the estimator
            ov_msckf::VioManager* manager;
            bool diverged;
            const FrameAlignment* alignment; // nullptr until the estimate is anchored
            const T* ground_truth_position; // episode-start frame
            const T* ground_truth_orientation;
            T position_error;
            T dead_reckoning_error;
        };

        FILE* pipe = nullptr;
        TI frames_written = 0;
        std::vector<std::array<T, 3>> ground_truth_trace, estimate_trace, map_points;
        T trace_min[2] = {0, 0}, trace_max[2] = {0, 0};
        bool has_estimate = false;
        T estimate_position[3], estimate_forward[3]; // episode-start frame
        std::vector<Landmark> landmarks;
        std::vector<Eigen::Vector3d> updated_landmarks; // consumed by the last MSCKF update
        bool update_ran = false;

        bool open(const std::string& path, T frame_rate){
            pipe = hyperdrone_demo::open_video_pipe(WIDTH, HEIGHT, (std::size_t)(frame_rate + (T)0.5), 1, path);
            return pipe != nullptr;
        }
        bool close(){
            if(pipe == nullptr){
                return false;
            }
            bool ok = hyperdrone_demo::close_video_pipe(pipe);
            pipe = nullptr;
            return ok;
        }
        void include_in_trace_bounds(const T position[3]){
            for(TI dim_i = 0; dim_i < 2; dim_i++){
                trace_min[dim_i] = std::min(trace_min[dim_i], position[dim_i]);
                trace_max[dim_i] = std::max(trace_max[dim_i], position[dim_i]);
            }
        }
        void record_ground_truth(const T position[3]){
            if(pipe == nullptr){
                return;
            }
            ground_truth_trace.push_back({position[0], position[1], position[2]});
            include_in_trace_bounds(position);
        }
        void record_estimate(const T position[3], const T forward[3]){
            if(pipe == nullptr){
                return;
            }
            estimate_trace.push_back({position[0], position[1], position[2]});
            include_in_trace_bounds(position);
            for(TI dim_i = 0; dim_i < 3; dim_i++){
                estimate_position[dim_i] = position[dim_i];
                estimate_forward[dim_i] = forward[dim_i];
            }
            has_estimate = true;
        }

        static cv::Point scaled(double x, double y){
            return cv::Point((int)std::lround(x * (double)SCALE), (int)std::lround(y * (double)SCALE));
        }
        static cv::Scalar depth_color(double depth){
            double t = (depth - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR);
            t = t < 0 ? 0 : (t > 1 ? 1 : t);
            // hue sweep red (near) -> yellow -> green -> cyan -> blue (far)
            double hue = t * 4.0;
            int sector = hue >= 4.0 ? 3 : (int)hue;
            double rising = 255.0 * (hue - sector);
            double falling = 255.0 - rising;
            switch(sector){
                case 0: return cv::Scalar(0, rising, 255);
                case 1: return cv::Scalar(0, 255, falling);
                case 2: return cv::Scalar(rising, 255, 0);
                default: return cv::Scalar(255, falling, 0);
            }
        }
        static void label(cv::Mat& image, const std::string& text, cv::Point origin, const cv::Scalar& color = COLOR_TEXT){
            cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_OUTLINE, 3, cv::LINE_AA);
            cv::putText(image, text, origin, cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
        }
        static bool project(const ov_msckf::State& state, const Eigen::Matrix3d& R_GtoI, const Eigen::Vector3d& p_IinG, const Eigen::Vector3d& p_FinG, cv::Point2f& pixel){
            const Eigen::Matrix3d R_ItoC = state._calib_IMUtoCAM.at(0)->Rot();
            const Eigen::Vector3d p_IinC = state._calib_IMUtoCAM.at(0)->pos();
            const Eigen::Vector3d p_FinC = R_ItoC * (R_GtoI * (p_FinG - p_IinG)) + p_IinC;
            if(p_FinC(2) < 0.05){
                return false;
            }
            const Eigen::Vector2d uv_norm(p_FinC(0) / p_FinC(2), p_FinC(1) / p_FinC(2));
            const Eigen::Vector2d uv = state._cam_intrinsics_cameras.at(0)->distort_d(uv_norm);
            pixel = cv::Point2f((float)uv(0), (float)uv(1));
            return uv(0) >= 0 && uv(0) < (double)CAM_WIDTH && uv(1) >= 0 && uv(1) < (double)CAM_HEIGHT;
        }

        void collect_landmarks(const Frame& frame){
            landmarks.clear();
            updated_landmarks.clear();
            update_ran = false;
            if(frame.diverged || !frame.manager->initialized()){
                return;
            }
            double active_time;
            std::unordered_map<std::size_t, Eigen::Vector3d> positions, measurements;
            frame.manager->get_active_tracks(active_time, positions, measurements);
            // zero-velocity frames skip the update and leave the previous frame's tracks behind
            if(active_time != frame.time){
                return;
            }
            update_ran = true;
            auto state = frame.manager->get_state();
            for(const auto& [id, measurement] : measurements){
                landmarks.push_back({id, positions.at(id), measurement, state->_features_SLAM.find(id) != state->_features_SLAM.end()});
            }
            updated_landmarks = frame.manager->get_good_features_MSCKF();
            if(frame.alignment != nullptr){
                for(const auto& position : updated_landmarks){
                    if(map_points.size() >= MAX_MAP_POINTS){
                        break;
                    }
                    T position_global[3] = {(T)position(0), (T)position(1), (T)position(2)};
                    std::array<T, 3> position_start;
                    frame.alignment->apply(position_global, position_start.data());
                    map_points.push_back(position_start);
                }
            }
        }

        void draw_tracks(const Frame& frame, cv::Mat& panel){
            cv::Mat history;
            if(!frame.diverged){
                history = frame.manager->get_historical_viz_image();
            }
            if(history.empty()){
                cv::cvtColor(*frame.image, history, cv::COLOR_GRAY2BGR);
            }
            cv::resize(history, panel, panel.size(), 0, 0, cv::INTER_NEAREST);
            label(panel, "KLT tracks: red = current, white = history, green box = SLAM", cv::Point(8, (int)PANEL_HEIGHT - 10));
        }

        void draw_landmarks(const Frame& frame, cv::Mat& panel){
            cv::Mat color;
            cv::cvtColor(*frame.image, color, cv::COLOR_GRAY2BGR);
            cv::resize(color, panel, panel.size(), 0, 0, cv::INTER_NEAREST);
            TI slam_count = 0, updated_visible = 0;
            const bool initialized = !frame.diverged && frame.manager->initialized();
            if(initialized){
                auto state = frame.manager->get_state();
                const Eigen::Matrix3d R_GtoI = state->_imu->Rot();
                const Eigen::Vector3d p_IinG = state->_imu->pos();
                for(const auto& landmark : landmarks){
                    const cv::Scalar landmark_color = depth_color(landmark.measurement(2));
                    const cv::Point measured = scaled(landmark.measurement(0), landmark.measurement(1));
                    cv::Point2f projected;
                    if(project(*state, R_GtoI, p_IinG, landmark.position, projected)){
                        const cv::Point projected_scaled = scaled(projected.x, projected.y);
                        cv::line(panel, measured, projected_scaled, landmark_color, 1, cv::LINE_AA);
                        cv::circle(panel, projected_scaled, 5, landmark_color, 1, cv::LINE_AA);
                    }
                    cv::circle(panel, measured, 3, landmark_color, cv::FILLED, cv::LINE_AA);
                    if(landmark.slam){
                        cv::rectangle(panel, measured - cv::Point(7, 7), measured + cv::Point(7, 7), COLOR_SLAM, 1, cv::LINE_AA);
                        slam_count++;
                    }
                }
                for(const auto& position : updated_landmarks){
                    cv::Point2f projected;
                    if(project(*state, R_GtoI, p_IinG, position, projected)){
                        cv::drawMarker(panel, scaled(projected.x, projected.y), COLOR_MSCKF, cv::MARKER_TILTED_CROSS, 9, 1, cv::LINE_AA);
                        updated_visible++;
                    }
                }
            }
            char text[160];
            const char* status = frame.diverged ? "DIVERGED" : (!initialized ? "initializing" : (update_ran ? "tracking" : "stationary (zero-velocity update)"));
            std::snprintf(text, sizeof(text), "t = %.2f s  %s", frame.time, status);
            label(panel, text, cv::Point(8, 18), frame.diverged ? COLOR_WARNING : COLOR_TEXT);
            if(initialized){
                std::snprintf(text, sizeof(text), "landmarks: %lu active (%lu SLAM), %lu of %lu MSCKF-updated in view", (unsigned long)landmarks.size(), (unsigned long)slam_count, (unsigned long)updated_visible, (unsigned long)updated_landmarks.size());
                label(panel, text, cv::Point(8, 36));
            }
            label(panel, "dot = tracked pixel, ring = landmark reprojection (hue = depth 0.5-8 m)", cv::Point(8, (int)PANEL_HEIGHT - 26));
            label(panel, "magenta x = landmark used by the last MSCKF update", cv::Point(8, (int)PANEL_HEIGHT - 10));
        }

        void draw_trace(const Frame& frame, cv::Mat& panel){
            panel.setTo(COLOR_BACKGROUND);
            const T extent_x = std::max(trace_max[0] - trace_min[0] + 2 * TRACE_MARGIN, TRACE_MIN_EXTENT);
            const T extent_y = std::max(trace_max[1] - trace_min[1] + 2 * TRACE_MARGIN, TRACE_MIN_EXTENT);
            const T pixels_per_meter = std::min((T)(TRACE_PANEL_WIDTH - 2 * TRACE_BORDER) / extent_y, (T)(PANEL_HEIGHT - 2 * TRACE_BORDER) / extent_x);
            const T center[2] = {(trace_min[0] + trace_max[0]) / 2, (trace_min[1] + trace_max[1]) / 2};
            // top-down view of the FLU episode-start frame: +x (forward) up, +y (left) left
            auto to_pixel = [&](const T* position){
                return cv::Point((int)std::lround((double)TRACE_PANEL_WIDTH / 2 - (double)(position[1] - center[1]) * pixels_per_meter), (int)std::lround((double)PANEL_HEIGHT / 2 - (double)(position[0] - center[0]) * pixels_per_meter));
            };
            const T half_span_x = (T)PANEL_HEIGHT / 2 / pixels_per_meter;
            const T half_span_y = (T)TRACE_PANEL_WIDTH / 2 / pixels_per_meter;
            for(int meter = (int)std::floor(center[1] - half_span_y); meter <= (int)std::ceil(center[1] + half_span_y); meter++){
                T position[3] = {center[0], (T)meter, 0};
                int u = to_pixel(position).x;
                cv::line(panel, cv::Point(u, 0), cv::Point(u, (int)PANEL_HEIGHT), COLOR_GRID, 1);
            }
            for(int meter = (int)std::floor(center[0] - half_span_x); meter <= (int)std::ceil(center[0] + half_span_x); meter++){
                T position[3] = {(T)meter, center[1], 0};
                int v = to_pixel(position).y;
                cv::line(panel, cv::Point(0, v), cv::Point((int)TRACE_PANEL_WIDTH, v), COLOR_GRID, 1);
            }
            for(const auto& point : map_points){
                cv::Point pixel = to_pixel(point.data());
                if(pixel.x >= 0 && pixel.x < (int)TRACE_PANEL_WIDTH && pixel.y >= 0 && pixel.y < (int)PANEL_HEIGHT){
                    panel.at<cv::Vec3b>(pixel) = cv::Vec3b((uchar)COLOR_MAP[0], (uchar)COLOR_MAP[1], (uchar)COLOR_MAP[2]);
                }
            }
            if(frame.alignment != nullptr){
                for(const auto& landmark : landmarks){
                    T position_global[3] = {(T)landmark.position(0), (T)landmark.position(1), (T)landmark.position(2)};
                    T position_start[3];
                    frame.alignment->apply(position_global, position_start);
                    cv::Point pixel = to_pixel(position_start);
                    if(landmark.slam){
                        cv::rectangle(panel, pixel - cv::Point(3, 3), pixel + cv::Point(3, 3), COLOR_SLAM, cv::FILLED);
                    } else {
                        cv::circle(panel, pixel, 1, depth_color(landmark.measurement(2)), cv::FILLED);
                    }
                }
            }
            auto draw_trajectory = [&](const std::vector<std::array<T, 3>>& trace, const cv::Scalar& color){
                std::vector<cv::Point> pixels;
                pixels.reserve(trace.size());
                for(const auto& point : trace){
                    pixels.push_back(to_pixel(point.data()));
                }
                cv::polylines(panel, pixels, false, color, 1, cv::LINE_AA);
            };
            auto draw_pose = [&](const T* position, const T* forward, const cv::Scalar& color){
                cv::Point pixel = to_pixel(position);
                double norm = std::sqrt((double)forward[0] * forward[0] + (double)forward[1] * forward[1]);
                if(norm > 1e-3){
                    cv::Point tip(pixel.x - (int)std::lround(14 * forward[1] / norm), pixel.y - (int)std::lround(14 * forward[0] / norm));
                    cv::line(panel, pixel, tip, color, 2, cv::LINE_AA);
                }
                cv::circle(panel, pixel, 3, color, cv::FILLED, cv::LINE_AA);
            };
            draw_trajectory(ground_truth_trace, COLOR_GROUND_TRUTH);
            if(has_estimate){
                draw_trajectory(estimate_trace, COLOR_ESTIMATE);
            }
            T ground_truth_forward[3];
            const T forward_axis[3] = {1, 0, 0};
            rlt::rl::environments::l2f::rotate_vector_by_quaternion<DEVICE, T>(frame.ground_truth_orientation, forward_axis, ground_truth_forward);
            draw_pose(frame.ground_truth_position, ground_truth_forward, COLOR_GROUND_TRUTH);
            if(has_estimate){
                draw_pose(estimate_position, estimate_forward, COLOR_ESTIMATE);
            }
            const int scale_bar_length = (int)std::lround(pixels_per_meter);
            const cv::Point scale_bar_origin(12, (int)PANEL_HEIGHT - 44);
            cv::line(panel, scale_bar_origin, scale_bar_origin + cv::Point(scale_bar_length, 0), COLOR_TEXT, 2);
            label(panel, "1 m", scale_bar_origin + cv::Point(scale_bar_length + 6, 4));
            char text[160];
            if(has_estimate){
                std::snprintf(text, sizeof(text), "position error: OpenVINS %.2f m, dead reckoning %.2f m", (double)frame.position_error, (double)frame.dead_reckoning_error);
            } else {
                std::snprintf(text, sizeof(text), "position error: dead reckoning %.2f m", (double)frame.dead_reckoning_error);
            }
            label(panel, text, cv::Point(8, 18));
            label(panel, "top-down: white = ground truth, orange = OpenVINS", cv::Point(8, (int)PANEL_HEIGHT - 26));
            label(panel, "gray = MSCKF map points, green = SLAM landmarks", cv::Point(8, (int)PANEL_HEIGHT - 10));
        }

        void write(const Frame& frame){
            if(pipe == nullptr){
                return;
            }
            collect_landmarks(frame);
            cv::Mat canvas((int)HEIGHT, (int)WIDTH, CV_8UC3);
            cv::Mat track_panel = canvas(cv::Rect(0, 0, (int)PANEL_WIDTH, (int)PANEL_HEIGHT));
            cv::Mat landmark_panel = canvas(cv::Rect((int)PANEL_WIDTH, 0, (int)PANEL_WIDTH, (int)PANEL_HEIGHT));
            cv::Mat trace_panel = canvas(cv::Rect((int)(2 * PANEL_WIDTH), 0, (int)TRACE_PANEL_WIDTH, (int)PANEL_HEIGHT));
            draw_tracks(frame, track_panel);
            draw_landmarks(frame, landmark_panel);
            draw_trace(frame, trace_panel);
            cv::line(canvas, cv::Point((int)PANEL_WIDTH, 0), cv::Point((int)PANEL_WIDTH, (int)HEIGHT), COLOR_TEXT_OUTLINE, 2);
            cv::line(canvas, cv::Point((int)(2 * PANEL_WIDTH), 0), cv::Point((int)(2 * PANEL_WIDTH), (int)HEIGHT), COLOR_TEXT_OUTLINE, 2);
            cv::Mat rgb;
            cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
            std::fwrite(rgb.data, 1, rgb.total() * rgb.elemSize(), pipe);
            frames_written++;
        }
    };
}
