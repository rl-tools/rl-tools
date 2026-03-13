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

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
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
static constexpr TI CAM_WIDTH = 64;
static constexpr TI CAM_HEIGHT = 64;

using CAPABILITY = rlt::nn::capability::Forward<>;
using MODEL = rlt::rendering::raytracing::yaw_prediction::MODEL<CAPABILITY, TYPE_POLICY, TI, BATCH_SIZE>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, BATCH_SIZE, CAM_HEIGHT, CAM_WIDTH, 3>;
using INPUT_SPEC = rlt::tensor::Specification<T, TI, INPUT_SHAPE>;
using OUTPUT_SHAPE = typename MODEL::OUTPUT_SHAPE;
using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;

static constexpr double PI = 3.14159265358979323846;

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
    std::array<double, 16> v;

    double& operator()(TI row, TI col) {
        return v[row * 4 + col];
    }

    double operator()(TI row, TI col) const {
        return v[row * 4 + col];
    }
};

struct FrameRecord {
    TI index;
    fs::path image_path;
    double timestamp;
    double translation_drift_meters;
    Mat4 world_transform;
    std::vector<float> pixels;
};

struct TargetAngles {
    double horizontal_norm;
    double vertical_norm;
    double roll_norm;
    double horizontal_degrees;
    double vertical_degrees;
    double roll_degrees;
};

struct Metrics {
    TI datasets = 0;
    TI frames = 0;
    TI evaluated_pairs = 0;
    double mae_horizontal_norm = 0.0;
    double mae_vertical_norm = 0.0;
    double mae_roll_norm = 0.0;
    double mae_horizontal_deg = 0.0;
    double mae_vertical_deg = 0.0;
    double mae_roll_deg = 0.0;
    double rmse_total_deg = 0.0;
    double max_horizontal_deg = 0.0;
    double max_vertical_deg = 0.0;
    double max_roll_deg = 0.0;
};

struct PairPrediction {
    Mat4 rel_ab;
    std::array<double, 3> prediction_degrees;
};

struct Convention {
    std::string name;
    bool reverse_pair = false;
    bool apply_training_basis = false;
    bool transpose_matrix = false;
    bool use_simple_roll = false;
    bool swap_hv = false;
    double sign_h = 1.0;
    double sign_v = 1.0;
    double sign_roll = 1.0;
};

static bool load_from_tar(ModelHandle& handle, const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "Failed to open model file: " << path << std::endl;
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
    bool success = rlt::load(handle.device, handle.model, model_group);
    fclose(f);
    return success;
}

static bool load_from_hdf5(ModelHandle& handle, const std::string& path) {
    try {
        auto file = HighFive::File(path, HighFive::File::ReadOnly);
        auto model_group = rlt::get_group(handle.device, file, "model");
        return rlt::load(handle.device, handle.model, model_group);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load HDF5 model: " << e.what() << std::endl;
        return false;
    }
}

static Mat4 parse_matrix(const nlohmann::json& json_matrix) {
    Mat4 matrix{};
    for (TI row = 0; row < 4; row++) {
        for (TI col = 0; col < 4; col++) {
            matrix(row, col) = json_matrix.at(row).at(col).get<double>();
        }
    }
    return matrix;
}

static Mat4 rigid_inverse(const Mat4& transform) {
    Mat4 inverse{};
    for (TI row = 0; row < 3; row++) {
        for (TI col = 0; col < 3; col++) {
            inverse(row, col) = transform(col, row);
        }
    }
    for (TI row = 0; row < 3; row++) {
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
    Mat4 result{};
    for (TI row = 0; row < 4; row++) {
        for (TI col = 0; col < 4; col++) {
            double sum = 0.0;
            for (TI k = 0; k < 4; k++) {
                sum += a(row, k) * b(k, col);
            }
            result(row, col) = sum;
        }
    }
    return result;
}

static Mat4 training_basis_transform() {
    Mat4 basis{};
    basis(0, 0) = 1.0;
    basis(1, 1) = 1.0;
    basis(2, 2) = -1.0;
    basis(3, 3) = 1.0;
    return basis;
}

static Mat4 transpose(const Mat4& matrix) {
    Mat4 result{};
    for (TI row = 0; row < 4; row++) {
        for (TI col = 0; col < 4; col++) {
            result(row, col) = matrix(col, row);
        }
    }
    return result;
}

static double normalized_displacement_to_degrees(double value, double fov_degrees) {
    const double half_fov_radians = fov_degrees * PI / 360.0;
    return std::atan(value * std::tan(half_fov_radians)) * 180.0 / PI;
}

static TargetAngles compute_ground_truth(const Mat4& reference, const Mat4& current, double horizontal_fov_degrees) {
    const Mat4 rel = mul(rigid_inverse(reference), current);
    const double z2x = -rel(0, 2);
    const double z2y = -rel(1, 2);
    const double z2z = -rel(2, 2);
    const double tan_half_h = std::tan(horizontal_fov_degrees * PI / 360.0);
    const double horizontal_norm = -z2y / (z2z * tan_half_h);
    const double vertical_norm = z2x / (z2z * tan_half_h);
    const double roll_degrees = -std::atan2(rel(0, 1), rel(0, 0)) * 180.0 / PI;
    return {
        horizontal_norm,
        vertical_norm,
        roll_degrees / 180.0,
        normalized_displacement_to_degrees(horizontal_norm, horizontal_fov_degrees),
        normalized_displacement_to_degrees(vertical_norm, horizontal_fov_degrees),
        roll_degrees
    };
}

static TargetAngles compute_ground_truth_variant(const Mat4& rel_ab, const Convention& convention, double horizontal_fov_degrees) {
    Mat4 rel = convention.reverse_pair ? rigid_inverse(rel_ab) : rel_ab;
    if (convention.apply_training_basis) {
        const Mat4 basis = training_basis_transform();
        rel = mul(mul(basis, rel), basis);
    }
    if (convention.transpose_matrix) {
        rel = transpose(rel);
    }

    const double z2x = rel(0, 2);
    const double z2y = rel(1, 2);
    const double z2z = rel(2, 2);
    const double tan_half_h = std::tan(horizontal_fov_degrees * PI / 360.0);
    const double px = normalized_displacement_to_degrees(z2x / (z2z * tan_half_h), horizontal_fov_degrees);
    const double py = normalized_displacement_to_degrees(z2y / (z2z * tan_half_h), horizontal_fov_degrees);

    double horizontal = px;
    double vertical = py;
    if (convention.swap_hv) {
        std::swap(horizontal, vertical);
    }
    horizontal *= convention.sign_h;
    vertical *= convention.sign_v;

    double roll_degrees = 0.0;
    if (convention.use_simple_roll) {
        roll_degrees = std::atan2(rel(1, 0), rel(0, 0)) * 180.0 / PI;
    } else {
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
        roll_degrees = std::atan2(re10, re00) * 180.0 / PI;
    }
    roll_degrees *= convention.sign_roll;

    return {horizontal, vertical, roll_degrees};
}

static double translation_norm(const Mat4& relative) {
    const double x = relative(0, 3);
    const double y = relative(1, 3);
    const double z = relative(2, 3);
    return std::sqrt(x * x + y * y + z * z);
}

static double relative_rotation_angle_degrees(const Mat4& relative) {
    const double trace = relative(0, 0) + relative(1, 1) + relative(2, 2);
    const double cosine = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosine) * 180.0 / PI;
}

static std::vector<float> load_image_64(const fs::path& path) {
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* input = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (!input) {
        std::ostringstream oss;
        oss << "Failed to load image: " << path << " (" << stbi_failure_reason() << ")";
        throw std::runtime_error(oss.str());
    }

    std::vector<unsigned char> resized(CAM_WIDTH * CAM_HEIGHT * 3);
    stbir_resize_uint8_linear(
        input, width, height, 0,
        resized.data(), CAM_WIDTH, CAM_HEIGHT, 0,
        STBIR_RGB
    );
    stbi_image_free(input);

    std::vector<float> output(CAM_WIDTH * CAM_HEIGHT * 3);
    for (TI i = 0; i < static_cast<TI>(output.size()); i++) {
        output[i] = static_cast<float>(resized[i]) / 255.0f;
    }
    return output;
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
        frame.index = json.at("index").get<TI>();
        frame.image_path = dataset_dir / json.at("image").get<std::string>();
        frame.timestamp = json.at("timestamp").get<double>();
        frame.translation_drift_meters = json.at("translationDriftMeters").get<double>();
        frame.world_transform = parse_matrix(json.at("worldTransform"));
        frame.pixels = load_image_64(frame.image_path);
        frames.push_back(std::move(frame));
    }
    return frames;
}

static std::array<double, 3> evaluate_pair(ModelHandle& handle, const FrameRecord& a, const FrameRecord& b) {
    std::memcpy(rlt::data(handle.input_a), a.pixels.data(), a.pixels.size() * sizeof(float));
    std::memcpy(rlt::data(handle.input_b), b.pixels.data(), b.pixels.size() * sizeof(float));

    auto mode = rlt::Mode<rlt::mode::Evaluation<>>{};
    rlt::evaluate(handle.device, handle.model, handle.input_a, handle.input_b, handle.output, handle.buffer, handle.rng, mode);

    const double pred_horizontal = normalized_displacement_to_degrees(rlt::get(handle.device, handle.output, 0, 0), 65.0);
    const double pred_vertical = normalized_displacement_to_degrees(rlt::get(handle.device, handle.output, 0, 1), 65.0);
    const double pred_roll = rlt::get(handle.device, handle.output, 0, 2) * 180.0;
    return {pred_horizontal, pred_vertical, pred_roll};
}

int main(int argc, char** argv) {
    std::vector<std::string> dataset_paths;
    std::string model_path = "tests/data/yaw-predictor5-beta.tar";
    double max_relative_translation_meters = 0.05;
    double min_relative_rotation_degrees = 1.0;
    double horizontal_fov_degrees = 65.0;
    double max_target_angle_degrees = 30.0;
    bool search_conventions = false;
    int anchor_index = -1;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            dataset_paths.push_back(argv[++i]);
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--max-relative-translation" && i + 1 < argc) {
            max_relative_translation_meters = std::atof(argv[++i]);
        } else if (arg == "--min-relative-rotation-deg" && i + 1 < argc) {
            min_relative_rotation_degrees = std::atof(argv[++i]);
        } else if (arg == "--fov" && i + 1 < argc) {
            horizontal_fov_degrees = std::atof(argv[++i]);
        } else if (arg == "--max-target-angle-deg" && i + 1 < argc) {
            max_target_angle_degrees = std::atof(argv[++i]);
        } else if (arg == "--search-conventions") {
            search_conventions = true;
        } else if (arg == "--anchor-index" && i + 1 < argc) {
            anchor_index = std::atoi(argv[++i]);
        } else {
            std::cerr << "Usage: " << argv[0]
                      << " --dataset <dir> [--dataset <dir> ...]"
                      << " [--model <path/to/yaw-predictor4.tar>]"
                      << " [--max-relative-translation <meters>]"
                      << " [--min-relative-rotation-deg <degrees>]"
                      << " [--max-target-angle-deg <degrees>]"
                      << " [--anchor-index <frame index>]"
                      << " [--search-conventions]"
                      << " [--fov <degrees>]" << std::endl;
            return 1;
        }
    }

    if (dataset_paths.empty()) {
        std::cerr << "Error: at least one --dataset is required" << std::endl;
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
    const bool loaded = is_hdf5 ? load_from_hdf5(handle, model_path) : load_from_tar(handle, model_path);
    if (!loaded) {
        std::cerr << "Failed to load model from: " << model_path << std::endl;
        return 1;
    }

    Metrics metrics;
    std::vector<PairPrediction> all_pairs;

    for (const auto& dataset_path : dataset_paths) {
        const fs::path dataset_dir = dataset_path;
        auto frames = load_dataset(dataset_dir);
        metrics.datasets += 1;
        metrics.frames += static_cast<TI>(frames.size());

        std::cout << "Dataset: " << dataset_dir << " (" << frames.size() << " frames)" << std::endl;

        TI dataset_pairs = 0;
        for (TI i = 0; i < static_cast<TI>(frames.size()); i++) {
            for (TI j = i + 1; j < static_cast<TI>(frames.size()); j++) {
                if (anchor_index >= 0) {
                    const TI anchor = static_cast<TI>(anchor_index);
                    if (i != anchor) {
                        continue;
                    }
                }
                const Mat4 relative = mul(rigid_inverse(frames[i].world_transform), frames[j].world_transform);
                const double relative_translation = translation_norm(relative);
                if (relative_translation > max_relative_translation_meters) {
                    continue;
                }
                const double relative_rotation_degrees = relative_rotation_angle_degrees(relative);
                if (relative_rotation_degrees < min_relative_rotation_degrees) {
                    continue;
                }

                const auto target = compute_ground_truth(frames[i].world_transform, frames[j].world_transform, horizontal_fov_degrees);
                if (std::abs(target.horizontal_degrees) > max_target_angle_degrees ||
                    std::abs(target.vertical_degrees) > max_target_angle_degrees ||
                    std::abs(target.roll_degrees) > max_target_angle_degrees) {
                    continue;
                }

                const auto prediction = evaluate_pair(handle, frames[i], frames[j]);
                if (search_conventions) {
                    all_pairs.push_back({relative, prediction});
                }

                const double prediction_horizontal_norm = std::tan(prediction[0] * PI / 180.0) /
                                                         std::tan(horizontal_fov_degrees * PI / 360.0);
                const double prediction_vertical_norm = std::tan(prediction[1] * PI / 180.0) /
                                                       std::tan(horizontal_fov_degrees * PI / 360.0);
                const double prediction_roll_norm = prediction[2] / 180.0;

                const double err_h_norm = std::abs(prediction_horizontal_norm - target.horizontal_norm);
                const double err_v_norm = std::abs(prediction_vertical_norm - target.vertical_norm);
                const double err_r_norm = std::abs(prediction_roll_norm - target.roll_norm);
                const double err_h = std::abs(prediction[0] - target.horizontal_degrees);
                const double err_v = std::abs(prediction[1] - target.vertical_degrees);
                const double err_r = std::abs(prediction[2] - target.roll_degrees);

                metrics.mae_horizontal_norm += err_h_norm;
                metrics.mae_vertical_norm += err_v_norm;
                metrics.mae_roll_norm += err_r_norm;
                metrics.mae_horizontal_deg += err_h;
                metrics.mae_vertical_deg += err_v;
                metrics.mae_roll_deg += err_r;
                metrics.rmse_total_deg += err_h * err_h + err_v * err_v + err_r * err_r;
                metrics.max_horizontal_deg = std::max(metrics.max_horizontal_deg, err_h);
                metrics.max_vertical_deg = std::max(metrics.max_vertical_deg, err_v);
                metrics.max_roll_deg = std::max(metrics.max_roll_deg, err_r);
                metrics.evaluated_pairs += 1;
                dataset_pairs += 1;
            }
        }
        std::cout << "  Evaluated pairs: " << dataset_pairs << std::endl;
    }

    if (metrics.evaluated_pairs == 0) {
        std::cerr << "No valid pairs were evaluated. Loosen the pair filters." << std::endl;
        return 1;
    }

    const double pair_count = static_cast<double>(metrics.evaluated_pairs);
    metrics.mae_horizontal_norm /= pair_count;
    metrics.mae_vertical_norm /= pair_count;
    metrics.mae_roll_norm /= pair_count;
    metrics.mae_horizontal_deg /= pair_count;
    metrics.mae_vertical_deg /= pair_count;
    metrics.mae_roll_deg /= pair_count;
    metrics.rmse_total_deg = std::sqrt(metrics.rmse_total_deg / (pair_count * 3.0));

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Real-world yaw prediction evaluation" << std::endl;
    std::cout << "  Model: " << model_path << std::endl;
    std::cout << "  Datasets: " << metrics.datasets << std::endl;
    std::cout << "  Frames: " << metrics.frames << std::endl;
    std::cout << "  Evaluated pairs: " << metrics.evaluated_pairs << std::endl;
    std::cout << "  Pair filter: translation <= " << max_relative_translation_meters
              << " m, rotation >= " << min_relative_rotation_degrees
              << " deg, |target| <= " << max_target_angle_degrees << " deg" << std::endl;
    std::cout << "  Train-style MAE err_px:   " << metrics.mae_horizontal_norm << std::endl;
    std::cout << "  Train-style MAE err_py:   " << metrics.mae_vertical_norm << std::endl;
    std::cout << "  Train-style MAE err_roll: " << metrics.mae_roll_norm << std::endl;
    if (anchor_index >= 0) {
        std::cout << "  Anchor index:   " << anchor_index << std::endl;
    }
    std::cout << "  MAE horizontal: " << metrics.mae_horizontal_deg << " deg" << std::endl;
    std::cout << "  MAE vertical:   " << metrics.mae_vertical_deg << " deg" << std::endl;
    std::cout << "  MAE roll:       " << metrics.mae_roll_deg << " deg" << std::endl;
    std::cout << "  RMSE total:     " << metrics.rmse_total_deg << " deg" << std::endl;
    std::cout << "  Max horizontal: " << metrics.max_horizontal_deg << " deg" << std::endl;
    std::cout << "  Max vertical:   " << metrics.max_vertical_deg << " deg" << std::endl;
    std::cout << "  Max roll:       " << metrics.max_roll_deg << " deg" << std::endl;

    if (search_conventions) {
        std::vector<Convention> conventions;
        for (bool reverse_pair : {false, true}) {
            for (bool apply_training_basis : {false, true}) {
                for (bool transpose_matrix : {false, true}) {
                    for (bool use_simple_roll : {false, true}) {
                        for (bool swap_hv : {false, true}) {
                            for (double sign_h : {-1.0, 1.0}) {
                                for (double sign_v : {-1.0, 1.0}) {
                                    for (double sign_roll : {-1.0, 1.0}) {
                                        Convention convention;
                                        convention.reverse_pair = reverse_pair;
                                        convention.apply_training_basis = apply_training_basis;
                                        convention.transpose_matrix = transpose_matrix;
                                        convention.use_simple_roll = use_simple_roll;
                                        convention.swap_hv = swap_hv;
                                        convention.sign_h = sign_h;
                                        convention.sign_v = sign_v;
                                        convention.sign_roll = sign_roll;
                                        std::ostringstream name;
                                        name << (reverse_pair ? "BA" : "AB")
                                             << (apply_training_basis ? "+basis" : "+raw")
                                             << (transpose_matrix ? "+T" : "")
                                             << (use_simple_roll ? "+simpleRoll" : "+trainRoll")
                                             << (swap_hv ? "+swapHV" : "")
                                             << (sign_h < 0 ? "+negH" : "+posH")
                                             << (sign_v < 0 ? "+negV" : "+posV")
                                             << (sign_roll < 0 ? "+negR" : "+posR");
                                        convention.name = name.str();
                                        conventions.push_back(convention);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        struct ConventionResult {
            std::string name;
            TI evaluated_pairs = 0;
            double mae_h = 0.0;
            double mae_v = 0.0;
            double mae_r = 0.0;
            double rmse = 0.0;
        };
        std::vector<ConventionResult> results;
        for (const auto& convention : conventions) {
            ConventionResult result;
            result.name = convention.name;
            for (const auto& pair : all_pairs) {
                const auto target = compute_ground_truth_variant(pair.rel_ab, convention, horizontal_fov_degrees);
                if (std::abs(target.horizontal_degrees) > max_target_angle_degrees ||
                    std::abs(target.vertical_degrees) > max_target_angle_degrees ||
                    std::abs(target.roll_degrees) > max_target_angle_degrees) {
                    continue;
                }
                const double err_h = std::abs(pair.prediction_degrees[0] - target.horizontal_degrees);
                const double err_v = std::abs(pair.prediction_degrees[1] - target.vertical_degrees);
                const double err_r = std::abs(pair.prediction_degrees[2] - target.roll_degrees);
                result.mae_h += err_h;
                result.mae_v += err_v;
                result.mae_r += err_r;
                result.rmse += err_h * err_h + err_v * err_v + err_r * err_r;
                result.evaluated_pairs += 1;
            }
            if (result.evaluated_pairs > 0) {
                const double denom = static_cast<double>(result.evaluated_pairs);
                result.mae_h /= denom;
                result.mae_v /= denom;
                result.mae_r /= denom;
                result.rmse = std::sqrt(result.rmse / (denom * 3.0));
                results.push_back(result);
            }
        }

        std::sort(results.begin(), results.end(), [](const ConventionResult& a, const ConventionResult& b) {
            return (a.mae_h + a.mae_v + a.mae_r) < (b.mae_h + b.mae_v + b.mae_r);
        });

        std::cout << "Convention search (top 10 by total MAE)" << std::endl;
        const TI limit = std::min<TI>(10, static_cast<TI>(results.size()));
        for (TI i = 0; i < limit; i++) {
            const auto& result = results[i];
            std::cout << "  " << (i + 1) << ". " << result.name
                      << "  pairs=" << result.evaluated_pairs
                      << "  mae_h=" << result.mae_h
                      << "  mae_v=" << result.mae_v
                      << "  mae_r=" << result.mae_r
                      << "  rmse=" << result.rmse
                      << std::endl;
        }
    }

    return 0;
}
