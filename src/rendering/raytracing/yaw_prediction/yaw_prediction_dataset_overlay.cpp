#include <rl_tools/operations/cpu.h>

#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn_models/parallel/operations_generic.h>
#include <rl_tools/nn_models/operations_generic.h>

#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#include <rl_tools/persist/backends/tar/operations_posix.h>
#include <rl_tools/persist/backends/hdf5/operations_cpu.h>

#include <rl_tools/nn/parameters/persist.h>
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>
#include <rl_tools/nn_models/parallel/persist.h>

#include "model.h"

#include <nlohmann/json.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace rlt = rl_tools;

using T = float;
using DEVICE = rlt::devices::DefaultCPU;
using TI = DEVICE::index_t;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;

static constexpr TI BATCH_SIZE = 1;
static constexpr TI MODEL_SIZE = 64;

using CAPABILITY = rlt::nn::capability::Forward<>;
using MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, MODEL_SIZE, MODEL_SIZE, 3>;
using INPUT_SPEC = rlt::tensor::Specification<T, TI, INPUT_SHAPE>;
using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;

struct ModelHandle {
    DEVICE device;
    MODEL model;
    typename MODEL::template Buffer<true> buffer;
    rlt::Tensor<INPUT_SPEC> input_a;
    rlt::Tensor<INPUT_SPEC> input_b;
    rlt::Tensor<OUTPUT_SPEC> output;
    typename DEVICE::SPEC::RANDOM::ENGINE<> rng;
};

struct Mat4 {
    std::array<double, 16> v{};

    double& operator()(int row, int col) {
        return v[row * 4 + col];
    }

    double operator()(int row, int col) const {
        return v[row * 4 + col];
    }
};

struct FrameRecord {
    int index = 0;
    fs::path image_path;
    Mat4 world_transform;
    cv::Mat image;
    std::vector<float> model_pixels;
};

struct Target2D {
    double horizontal_norm = 0.0;
    double vertical_norm = 0.0;
};

static Mat4 parse_matrix(const nlohmann::json& json_matrix) {
    Mat4 matrix;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            matrix(row, col) = json_matrix.at(row).at(col).get<double>();
        }
    }
    return matrix;
}

static Mat4 rigid_inverse(const Mat4& transform) {
    Mat4 inverse;
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            inverse(row, col) = transform(col, row);
        }
    }
    for (int row = 0; row < 3; row++) {
        inverse(row, 3) =
            -(inverse(row, 0) * transform(0, 3) +
              inverse(row, 1) * transform(1, 3) +
              inverse(row, 2) * transform(2, 3));
    }
    inverse(3, 0) = 0.0;
    inverse(3, 1) = 0.0;
    inverse(3, 2) = 0.0;
    inverse(3, 3) = 1.0;
    return inverse;
}

static Mat4 mul(const Mat4& a, const Mat4& b) {
    Mat4 result;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            double sum = 0.0;
            for (int k = 0; k < 4; k++) {
                sum += a(row, k) * b(k, col);
            }
            result(row, col) = sum;
        }
    }
    return result;
}

static std::vector<FrameRecord> load_dataset(const fs::path& dataset_dir) {
    const fs::path metadata_path = dataset_dir / "frames.jsonl";
    std::ifstream input(metadata_path);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open metadata file: " + metadata_path.string());
    }

    std::vector<FrameRecord> frames;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto json = nlohmann::json::parse(line);
        FrameRecord frame;
        frame.index = json.at("index").get<int>();
        frame.image_path = dataset_dir / json.at("image").get<std::string>();
        frame.world_transform = parse_matrix(json.at("worldTransform"));
        frame.image = cv::imread(frame.image_path.string(), cv::IMREAD_COLOR);
        if (frame.image.empty()) {
            throw std::runtime_error("Failed to load frame image: " + frame.image_path.string());
        }
        cv::Mat resized_rgb;
        cv::resize(frame.image, resized_rgb, cv::Size(MODEL_SIZE, MODEL_SIZE), 0.0, 0.0, cv::INTER_LINEAR);
        frame.model_pixels.resize(MODEL_SIZE * MODEL_SIZE * 3);
        for (int y = 0; y < MODEL_SIZE; y++) {
            for (int x = 0; x < MODEL_SIZE; x++) {
                const cv::Vec3b bgr = resized_rgb.at<cv::Vec3b>(y, x);
                const int idx = (y * MODEL_SIZE + x) * 3;
                frame.model_pixels[idx + 0] = static_cast<float>(bgr[2]) / 255.0f;
                frame.model_pixels[idx + 1] = static_cast<float>(bgr[1]) / 255.0f;
                frame.model_pixels[idx + 2] = static_cast<float>(bgr[0]) / 255.0f;
            }
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

static bool load_model_from_tar(ModelHandle& handle, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        return false;
    }
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    rlt::persist::backends::tar::PosixFileData<TI> file_data;
    file_data.f = f;
    file_data.size = file_size;

    using READER_GROUP_SPEC = rlt::persist::backends::tar::ReaderGroupSpecification<TI, rlt::persist::backends::tar::PosixFileData<TI>>;
    rlt::persist::backends::tar::ReaderGroup<READER_GROUP_SPEC> reader_group;
    reader_group.data = file_data;

    auto model_group = rlt::get_group(handle.device, reader_group, "model");
    const bool success = rlt::load(handle.device, handle.model, model_group);
    fclose(f);
    return success;
}

static bool load_model_from_hdf5(ModelHandle& handle, const std::string& path) {
    try {
        auto file = HighFive::File(path, HighFive::File::ReadOnly);
        auto model_group = rlt::get_group(handle.device, file, "model");
        return rlt::load(handle.device, handle.model, model_group);
    } catch (...) {
        return false;
    }
}

static std::array<double, 3> evaluate_pair(ModelHandle& handle, const FrameRecord& reference, const FrameRecord& current) {
    std::memcpy(rlt::data(handle.input_a), reference.model_pixels.data(), reference.model_pixels.size() * sizeof(float));
    std::memcpy(rlt::data(handle.input_b), current.model_pixels.data(), current.model_pixels.size() * sizeof(float));

    auto mode = rlt::Mode<rlt::mode::Evaluation<>>{};
    rlt::evaluate(handle.device, handle.model, handle.input_a, handle.input_b, handle.output, handle.buffer, handle.rng, mode);

    return {
        static_cast<double>(rlt::get(handle.device, handle.output, 0, 0)),
        static_cast<double>(rlt::get(handle.device, handle.output, 0, 1)),
        static_cast<double>(rlt::get(handle.device, handle.output, 0, 2))
    };
}

static Target2D compute_ground_truth_app(const Mat4& reference, const Mat4& current, double horizontal_fov_degrees) {
    const Mat4 rel = mul(rigid_inverse(reference), current);
    const double z2x = -rel(0, 2);
    const double z2y = -rel(1, 2);
    const double z2z = -rel(2, 2);
    const double tan_half_h = std::tan(horizontal_fov_degrees * 0.5 * CV_PI / 180.0);
    return {
        -z2y / (z2z * tan_half_h),
        z2x / (z2z * tan_half_h)
    };
}

static double compute_roll_shortest_arc(const Mat4& reference, const Mat4& current) {
    const Mat4 rel_arkit = mul(rigid_inverse(reference), current);

    Mat4 basis{};
    basis(0, 0) = 1.0;
    basis(1, 1) = 1.0;
    basis(2, 2) = -1.0;
    basis(3, 3) = 1.0;

    const Mat4 rel = mul(mul(basis, rel_arkit), basis);

    const double z2x = rel(0, 2);
    const double z2y = rel(1, 2);
    const double z2z = rel(2, 2);
    const double z2_len = std::sqrt(z2x * z2x + z2y * z2y + z2z * z2z);
    const double nx = z2x / z2_len;
    const double ny = z2y / z2_len;
    const double nz = z2z / z2_len;
    const double k = 1.0 / (1.0 + nz);

    const double eff00 = rel(0, 0);
    const double eff10 = rel(1, 0);
    const double eff20 = rel(2, 0);
    const double re00 = (1.0 - nx * nx * k) * eff00 + (-nx * ny * k) * eff10 + (-nx) * eff20;
    const double re10 = (-nx * ny * k) * eff00 + (1.0 - ny * ny * k) * eff10 + (-ny) * eff20;
    return std::atan2(re10, re00) * 180.0 / CV_PI;
}

static double normalized_displacement_to_degrees(double value, double fov_degrees) {
    const double half_fov_radians = fov_degrees * 0.5 * CV_PI / 180.0;
    return std::atan(value * std::tan(half_fov_radians)) * 180.0 / CV_PI;
}

static cv::Point target_to_pixel(const Target2D& target, int width, int height) {
    const double cx = (static_cast<double>(width) - 1.0) * 0.5;
    const double cy = (static_cast<double>(height) - 1.0) * 0.5;
    const double x = cx + target.horizontal_norm * cx;
    const double y = cy - target.vertical_norm * cy;
    return {
        static_cast<int>(std::round(x)),
        static_cast<int>(std::round(y))
    };
}

static void draw_cross(cv::Mat& image, const cv::Point& center, const cv::Scalar& color, int size, int thickness) {
    cv::line(image, center + cv::Point(-size, 0), center + cv::Point(size, 0), color, thickness, cv::LINE_AA);
    cv::line(image, center + cv::Point(0, -size), center + cv::Point(0, size), color, thickness, cv::LINE_AA);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0]
                  << " <dataset_dir>"
                  << " [--model <path.{h5,tar}>]"
                  << " [--output <path.mp4>]"
                  << " [--fps <value>]"
                  << " [--scale <integer>]"
                  << " [--fov <degrees>]" << std::endl;
        return 1;
    }

    fs::path dataset_dir = argv[1];
    std::string model_path = "tests/data/yaw-predictor5-beta.h5";
    fs::path output_path = dataset_dir / "overlay_markers.mp4";
    double fps = 12.0;
    int scale = -1;
    double horizontal_fov_degrees = 65.0;
    const int text_band_height = 118;

    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else if (arg == "--fps" && i + 1 < argc) {
            fps = std::stod(argv[++i]);
        } else if (arg == "--scale" && i + 1 < argc) {
            scale = std::stoi(argv[++i]);
        } else if (arg == "--fov" && i + 1 < argc) {
            horizontal_fov_degrees = std::stod(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            return 1;
        }
    }

    if (!fs::exists(dataset_dir)) {
        std::cerr << "Dataset directory does not exist: " << dataset_dir << std::endl;
        return 1;
    }

    const auto frames = load_dataset(dataset_dir);
    if (frames.empty()) {
        std::cerr << "Dataset is empty: " << dataset_dir << std::endl;
        return 1;
    }

    ModelHandle handle;
    rlt::malloc(handle.device, handle.rng);
    rlt::init(handle.device, handle.rng, 0);
    rlt::malloc(handle.device, handle.model);
    rlt::malloc(handle.device, handle.buffer);
    rlt::malloc(handle.device, handle.input_a);
    rlt::malloc(handle.device, handle.input_b);
    rlt::malloc(handle.device, handle.output);

    const bool is_hdf5 = model_path.size() >= 3 && model_path.substr(model_path.size() - 3) == ".h5";
    const bool loaded = is_hdf5 ? load_model_from_hdf5(handle, model_path) : load_model_from_tar(handle, model_path);
    if (!loaded) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    const cv::Mat& base = frames.front().image;

    if (scale <= 0) {
        scale = base.cols <= 128 ? 8 : 1;
    }

    const cv::Size panel_size(base.cols * scale, base.rows * scale);
    const cv::Size output_size(panel_size.width * 2, panel_size.height + text_band_height);
    const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
    cv::VideoWriter writer(output_path.string(), fourcc, fps, output_size);
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer: " << output_path << std::endl;
        return 1;
    }

    const Mat4 reference_transform = frames.front().world_transform;
    const cv::Point center((base.cols - 1) / 2, (base.rows - 1) / 2);

    for (const auto& frame : frames) {
        cv::Mat left = base.clone();
        cv::Mat right = frame.image.clone();

        const Target2D gt = compute_ground_truth_app(reference_transform, frame.world_transform, horizontal_fov_degrees);
        const auto pred = evaluate_pair(handle, frames.front(), frame);
        const Target2D pred_target{pred[0], pred[1]};
        const double pred_horizontal_deg = normalized_displacement_to_degrees(pred[0], horizontal_fov_degrees);
        const double pred_vertical_deg = normalized_displacement_to_degrees(pred[1], horizontal_fov_degrees);
        const double pred_roll_deg = pred[2] * 180.0;
        const double gt_horizontal_deg = normalized_displacement_to_degrees(gt.horizontal_norm, horizontal_fov_degrees);
        const double gt_vertical_deg = normalized_displacement_to_degrees(gt.vertical_norm, horizontal_fov_degrees);
        const double gt_roll_deg = compute_roll_shortest_arc(reference_transform, frame.world_transform);
        const double err_horizontal_deg = pred_horizontal_deg - gt_horizontal_deg;
        const double err_vertical_deg = pred_vertical_deg - gt_vertical_deg;
        const double err_roll_deg = pred_roll_deg - gt_roll_deg;

        const cv::Point gt_px = target_to_pixel(gt, left.cols, left.rows);
        const cv::Point pred_px = target_to_pixel(pred_target, left.cols, left.rows);

        draw_cross(left, center, cv::Scalar(180, 180, 180), 10, 1);
        draw_cross(left, gt_px, cv::Scalar(0, 255, 0), 10, 2);
        draw_cross(left, pred_px, cv::Scalar(0, 0, 255), 10, 2);
        draw_cross(right, center, cv::Scalar(255, 255, 255), 10, 2);

        cv::putText(left, "reference", cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(right, "current", cv::Point(8, 20), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(left, "GT", gt_px + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::putText(left, "Pred", pred_px + cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        cv::putText(right, "Center", center + cv::Point(8, -8), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
        cv::putText(left, "frame " + std::to_string(frame.index), cv::Point(8, left.rows - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);

        cv::Mat left_scaled, right_scaled;
        cv::resize(left, left_scaled, panel_size, 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(right, right_scaled, panel_size, 0.0, 0.0, cv::INTER_NEAREST);

        cv::Mat canvas(output_size, CV_8UC3, cv::Scalar(18, 18, 18));
        left_scaled.copyTo(canvas(cv::Rect(0, 0, panel_size.width, panel_size.height)));
        right_scaled.copyTo(canvas(cv::Rect(panel_size.width, 0, panel_size.width, panel_size.height)));
        cv::rectangle(
            canvas,
            cv::Rect(0, panel_size.height, output_size.width, text_band_height),
            cv::Scalar(245, 245, 245),
            cv::FILLED
        );
        cv::line(
            canvas,
            cv::Point(0, panel_size.height),
            cv::Point(output_size.width, panel_size.height),
            cv::Scalar(210, 210, 210),
            1,
            cv::LINE_AA
        );

        const int text_y1 = panel_size.height + 28;
        const int text_y2 = panel_size.height + 56;
        const int text_y3 = panel_size.height + 84;
        const int text_y4 = panel_size.height + 108;
        cv::putText(
            canvas,
            cv::format("H: %+0.1f deg  V: %+0.1f deg  R: %+0.1f deg", pred_horizontal_deg, pred_vertical_deg, pred_roll_deg),
            cv::Point(18, text_y1),
            cv::FONT_HERSHEY_DUPLEX,
            0.8,
            cv::Scalar(20, 20, 20),
            1,
            cv::LINE_AA
        );
        cv::putText(
            canvas,
            cv::format("GT H: %+0.1f deg  V: %+0.1f deg  R: %+0.1f deg", gt_horizontal_deg, gt_vertical_deg, gt_roll_deg),
            cv::Point(18, text_y2),
            cv::FONT_HERSHEY_DUPLEX,
            0.72,
            cv::Scalar(180, 90, 20),
            1,
            cv::LINE_AA
        );
        cv::putText(
            canvas,
            cv::format("Err H/V: %+0.1f deg  %+0.1f deg", err_horizontal_deg, err_vertical_deg),
            cv::Point(18, text_y3),
            cv::FONT_HERSHEY_DUPLEX,
            0.68,
            cv::Scalar(30, 50, 190),
            1,
            cv::LINE_AA
        );
        cv::putText(
            canvas,
            cv::format("Err R: %+0.1f deg  (GT uses shortest-arc roll)", err_roll_deg),
            cv::Point(18, text_y4),
            cv::FONT_HERSHEY_DUPLEX,
            0.62,
            cv::Scalar(80, 80, 80),
            1,
            cv::LINE_AA
        );
        writer.write(canvas);
    }

    writer.release();
    std::cout << "Wrote overlay video to " << output_path << std::endl;
    return 0;
}
