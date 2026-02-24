// ImageNet-1k training with ResNet-18, implementing the timm training recipe
// Reference: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
//
// Hyperparameters (timm resnet18.a1_in1k defaults):
//   Optimizer:  SGD, momentum=0.9, nesterov=true, weight_decay=2e-5
//   LR:         0.1 (base, scaled by batch_size/256), cosine schedule, 5-epoch warmup
//   Epochs:     300
//   Batch size: 128 (gradient accumulation with per-sample forward/backward)
//   Augmentation: random resize crop (scale 0.08-1.0, ratio 3/4-4/3), hflip p=0.5
//   Loss:       cross-entropy with label smoothing 0.1
//
// Reads ImageNet-1k directly from HuggingFace parquet files (Apache Arrow).

#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/operations_generic.h>
#include <rl_tools/nn/loss_functions/categorical_cross_entropy/operations_generic.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>

#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

#include <rl_tools/nn_models/resnet/resnet.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <filesystem>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

// ======================== Hyperparameters (timm resnet18.a1_in1k recipe) ========================

struct TrainingConfig {
    static constexpr TI IMAGE_SIZE = 224;
    static constexpr TI NUM_CLASSES = 1000;
    static constexpr TI NUM_EPOCHS = 300;
    static constexpr T BASE_LR = 0.1;
    static constexpr TI BASE_BATCH_SIZE = 256;
    static constexpr T WARMUP_LR = 1e-5;
    static constexpr TI WARMUP_EPOCHS = 5;
    static constexpr T MIN_LR = 0.0;
    static constexpr T MOMENTUM = 0.9;
    static constexpr T WEIGHT_DECAY = 2e-5;
    static constexpr bool NESTEROV = true;
    static constexpr T LABEL_SMOOTHING = 0.1;
    static constexpr T CROP_SCALE_MIN = 0.08;
    static constexpr T CROP_SCALE_MAX = 1.0;
    static constexpr T CROP_RATIO_MIN = 0.75;
    static constexpr T CROP_RATIO_MAX = 1.3333;
    static constexpr T HFLIP_PROB = 0.5;
    static constexpr TI CHECKPOINT_INTERVAL = 10;
};

// ======================== SGD optimizer configuration ========================

struct SGDParams: rlt::nn::optimizers::sgd::DefaultParameters<TYPE_POLICY>{
    static constexpr T LEARNING_RATE = TrainingConfig::BASE_LR;
    static constexpr T MOMENTUM = TrainingConfig::MOMENTUM;
    static constexpr T WEIGHT_DECAY = TrainingConfig::WEIGHT_DECAY;
    static constexpr bool NESTEROV = TrainingConfig::NESTEROV;
    static constexpr bool ENABLE_WEIGHT_DECAY = true;
};

using OPTIMIZER_SPEC = rlt::nn::optimizers::sgd::Specification<TYPE_POLICY, TI, SGDParams>;
using OPTIMIZER = rlt::nn::optimizers::SGD<OPTIMIZER_SPEC>;

// ======================== Model definition ========================

constexpr TI INTERNAL_BATCH_SIZE = 1;
using CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD, INTERNAL_BATCH_SIZE>;
using INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TrainingConfig::IMAGE_SIZE, TrainingConfig::IMAGE_SIZE, 3>;
using RESNET18 = rlt::nn_models::sequential::Build<CAPABILITY, rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI>, INPUT_SHAPE>;

// ======================== Parquet dataset reader ========================
// Reads HuggingFace ImageNet-1k parquet files directly.
// Schema: image: struct<bytes: binary, path: string>, label: int64

struct ParquetSample {
    std::vector<uint8_t> image_bytes;
    int64_t label;
};

struct ParquetShard {
    std::vector<ParquetSample> samples;

    bool load(const std::string& path) {
        auto maybe_file = arrow::io::ReadableFile::Open(path);
        if (!maybe_file.ok()) {
            std::cerr << "Failed to open: " << path << " (" << maybe_file.status().ToString() << ")" << std::endl;
            return false;
        }
        auto maybe_reader = parquet::arrow::OpenFile(*maybe_file, arrow::default_memory_pool());
        if (!maybe_reader.ok()) {
            std::cerr << "Failed to open parquet: " << path << " (" << maybe_reader.status().ToString() << ")" << std::endl;
            return false;
        }
        auto& reader = *maybe_reader;
        std::shared_ptr<arrow::Table> table;
        auto status = reader->ReadTable(&table);
        if (!status.ok()) {
            std::cerr << "Failed to read table: " << path << " (" << status.ToString() << ")" << std::endl;
            return false;
        }

        auto image_chunked = table->GetColumnByName("image");
        auto label_chunked = table->GetColumnByName("label");
        if (!image_chunked || !label_chunked) {
            std::cerr << "Missing image or label column in: " << path << std::endl;
            return false;
        }

        samples.clear();
        samples.reserve(table->num_rows());

        for (int chunk_i = 0; chunk_i < image_chunked->num_chunks(); chunk_i++) {
            auto image_array = std::static_pointer_cast<arrow::StructArray>(image_chunked->chunk(chunk_i));
            auto label_array = std::static_pointer_cast<arrow::Int64Array>(label_chunked->chunk(chunk_i));
            auto bytes_field = image_array->GetFieldByName("bytes");
            if (!bytes_field) {
                std::cerr << "Missing 'bytes' field in image struct in: " << path << std::endl;
                return false;
            }
            auto bytes_array = std::static_pointer_cast<arrow::BinaryArray>(bytes_field);

            for (int64_t row = 0; row < image_array->length(); row++) {
                if (bytes_array->IsNull(row) || label_array->IsNull(row)) continue;
                auto view = bytes_array->GetView(row);
                ParquetSample s;
                s.image_bytes.assign(
                    reinterpret_cast<const uint8_t*>(view.data()),
                    reinterpret_cast<const uint8_t*>(view.data()) + view.size());
                s.label = label_array->Value(row);
                samples.push_back(std::move(s));
            }
        }
        return true;
    }
};

std::vector<std::string> find_parquet_files(const std::string& base_dir, const std::string& split) {
    std::vector<std::string> files;
    std::string data_dir = base_dir + "/data";
    if (!fs::exists(data_dir)) {
        data_dir = base_dir;
    }
    for (auto& entry : fs::directory_iterator(data_dir)) {
        auto fname = entry.path().filename().string();
        if (fname.find(split + "-") == 0 && fname.find(".parquet") != std::string::npos
            && fname.find(".metadata") == std::string::npos) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

// ======================== Image preprocessing ========================

struct DecodedImage {
    std::vector<uint8_t> pixels;
    int width = 0, height = 0;
    bool valid = false;
};

DecodedImage decode_jpeg(const uint8_t* jpeg_bytes, size_t jpeg_size) {
    DecodedImage img;
    int channels;
    uint8_t* raw = stbi_load_from_memory(jpeg_bytes, static_cast<int>(jpeg_size), &img.width, &img.height, &channels, 3);
    if (!raw) return img;
    img.pixels.assign(raw, raw + img.width * img.height * 3);
    stbi_image_free(raw);
    img.valid = true;
    return img;
}

void random_resize_crop(const DecodedImage& src, uint8_t* dst, TI target_size, std::mt19937& rng) {
    std::uniform_real_distribution<T> scale_dist(TrainingConfig::CROP_SCALE_MIN, TrainingConfig::CROP_SCALE_MAX);
    std::uniform_real_distribution<T> ratio_dist(
        std::log(TrainingConfig::CROP_RATIO_MIN),
        std::log(TrainingConfig::CROP_RATIO_MAX));

    int crop_w, crop_h, crop_x, crop_y;
    bool found = false;

    for (int attempt = 0; attempt < 10; attempt++) {
        T area = static_cast<T>(src.width * src.height);
        T target_area = scale_dist(rng) * area;
        T log_ratio = ratio_dist(rng);
        T ratio = std::exp(log_ratio);

        crop_w = static_cast<int>(std::sqrt(target_area * ratio) + 0.5);
        crop_h = static_cast<int>(std::sqrt(target_area / ratio) + 0.5);

        if (crop_w > 0 && crop_w <= src.width && crop_h > 0 && crop_h <= src.height) {
            std::uniform_int_distribution<int> x_dist(0, src.width - crop_w);
            std::uniform_int_distribution<int> y_dist(0, src.height - crop_h);
            crop_x = x_dist(rng);
            crop_y = y_dist(rng);
            found = true;
            break;
        }
    }

    if (!found) {
        T img_ratio = static_cast<T>(src.width) / src.height;
        if (img_ratio < TrainingConfig::CROP_RATIO_MIN) {
            crop_w = src.width;
            crop_h = static_cast<int>(src.width / TrainingConfig::CROP_RATIO_MIN);
        } else if (img_ratio > TrainingConfig::CROP_RATIO_MAX) {
            crop_h = src.height;
            crop_w = static_cast<int>(src.height * TrainingConfig::CROP_RATIO_MAX);
        } else {
            crop_w = src.width;
            crop_h = src.height;
        }
        crop_w = std::min(crop_w, src.width);
        crop_h = std::min(crop_h, src.height);
        crop_x = (src.width - crop_w) / 2;
        crop_y = (src.height - crop_h) / 2;
    }

    std::vector<uint8_t> cropped(crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++) {
        std::memcpy(
            cropped.data() + y * crop_w * 3,
            src.pixels.data() + (crop_y + y) * src.width * 3 + crop_x * 3,
            crop_w * 3);
    }

    stbir_resize_uint8_linear(
        cropped.data(), crop_w, crop_h, 0,
        dst, target_size, target_size, 0,
        STBIR_RGB);
}

void center_crop_resize(const DecodedImage& src, uint8_t* dst, TI target_size) {
    constexpr int RESIZE_SIZE = 256;
    T scale = static_cast<T>(RESIZE_SIZE) / std::min(src.width, src.height);
    int resized_w = static_cast<int>(src.width * scale + 0.5);
    int resized_h = static_cast<int>(src.height * scale + 0.5);
    resized_w = std::max(resized_w, RESIZE_SIZE);
    resized_h = std::max(resized_h, RESIZE_SIZE);

    std::vector<uint8_t> resized(resized_w * resized_h * 3);
    stbir_resize_uint8_linear(
        src.pixels.data(), src.width, src.height, 0,
        resized.data(), resized_w, resized_h, 0,
        STBIR_RGB);

    int x_off = (resized_w - static_cast<int>(target_size)) / 2;
    int y_off = (resized_h - static_cast<int>(target_size)) / 2;
    for (TI y = 0; y < target_size; y++) {
        std::memcpy(
            dst + y * target_size * 3,
            resized.data() + (y_off + y) * resized_w * 3 + x_off * 3,
            target_size * 3);
    }
}

template<typename DEVICE_T, typename TENSOR>
void preprocess_to_tensor(DEVICE_T& device, const uint8_t* rgb_224, TENSOR& input_tensor) {
    constexpr TI SIZE = TrainingConfig::IMAGE_SIZE;
    using FLAT_SHAPE = rlt::tensor::Shape<TI, 1, SIZE, SIZE, 3>;
    auto input_4d = rlt::view_memory<FLAT_SHAPE>(device, input_tensor);
    for (TI h = 0; h < SIZE; h++) {
        for (TI w = 0; w < SIZE; w++) {
            for (TI c = 0; c < 3; c++) {
                T pixel = static_cast<T>(rgb_224[(h * SIZE + w) * 3 + c]) / 255.0f;
                T normalized = (pixel - static_cast<T>(rlt::nn_models::resnet18::IMAGENET_MEAN[c]))
                             / static_cast<T>(rlt::nn_models::resnet18::IMAGENET_STD[c]);
                rlt::set(device, input_4d, normalized, (TI)0, h, w, c);
            }
        }
    }
}

// ======================== Learning rate schedule ========================

T cosine_lr(TI epoch, TI total_epochs, T base_lr, T min_lr, TI warmup_epochs, T warmup_lr) {
    if (epoch < warmup_epochs) {
        return warmup_lr + (base_lr - warmup_lr) * static_cast<T>(epoch) / static_cast<T>(warmup_epochs);
    }
    T progress = static_cast<T>(epoch - warmup_epochs) / static_cast<T>(total_epochs - warmup_epochs);
    return min_lr + (base_lr - min_lr) * 0.5f * (1.0f + std::cos(static_cast<T>(M_PI) * progress));
}

// ======================== Cross-entropy with label smoothing ========================

T cross_entropy_with_smoothing(const T* logits, TI target, TI num_classes, T smoothing) {
    T max_logit = logits[0];
    for (TI i = 1; i < num_classes; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }
    T sum_exp = 0;
    for (TI i = 0; i < num_classes; i++) {
        sum_exp += std::exp(logits[i] - max_logit);
    }
    T log_sum_exp = max_logit + std::log(sum_exp);

    T nll = -(logits[target] - log_sum_exp);
    T kl_uniform = log_sum_exp;
    for (TI i = 0; i < num_classes; i++) {
        kl_uniform -= logits[i] / static_cast<T>(num_classes);
    }

    return (1.0f - smoothing) * nll + smoothing * kl_uniform;
}

void cross_entropy_gradient_with_smoothing(const T* logits, TI target, TI num_classes, T smoothing, T loss_weight, T* d_logits) {
    T max_logit = logits[0];
    for (TI i = 1; i < num_classes; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }
    T sum_exp = 0;
    for (TI i = 0; i < num_classes; i++) {
        sum_exp += std::exp(logits[i] - max_logit);
    }

    T smooth_weight = smoothing / static_cast<T>(num_classes);

    for (TI i = 0; i < num_classes; i++) {
        T softmax_i = std::exp(logits[i] - max_logit) / sum_exp;
        T smooth_target = (i == target) ? ((1.0f - smoothing) + smooth_weight) : smooth_weight;
        d_logits[i] = (softmax_i - smooth_target) * loss_weight;
    }
}

// ======================== Training step on one shard ========================

struct TrainStats {
    T loss_sum = 0;
    TI correct = 0;
    TI total = 0;
    TI failed_decodes = 0;
};

template<typename MODEL, typename BUFFER, typename OPT, typename RNG_T>
TrainStats train_on_shard(
    DEVICE& device, MODEL& model, BUFFER& buffer, OPT& optimizer,
    RNG_T& rng, std::mt19937& data_rng,
    ParquetShard& shard,
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>>& input_tensor,
    rlt::Tensor<rlt::tensor::Specification<T, TI, typename MODEL::OUTPUT_SHAPE>>& d_loss_tensor,
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>>& d_input_tensor,
    std::vector<uint8_t>& resized_img,
    TI batch_i_offset, TI total_batches,
    TI batch_size, TI log_interval
) {
    TrainStats stats;
    TI num_samples = shard.samples.size();

    std::vector<TI> indices(num_samples);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), data_rng);

    TI num_batches = num_samples / batch_size;
    rlt::Mode<rlt::mode::Default<>> train_mode;

    for (TI batch_i = 0; batch_i < num_batches; batch_i++) {
        auto batch_start = std::chrono::high_resolution_clock::now();
        rlt::zero_gradient(device, model);
        T batch_loss = 0;
        TI batch_correct = 0;
        TI batch_valid = 0;

        for (TI sample_i = 0; sample_i < batch_size; sample_i++) {
            TI idx = indices[batch_i * batch_size + sample_i];
            auto& s = shard.samples[idx];

            auto decoded = decode_jpeg(s.image_bytes.data(), s.image_bytes.size());
            if (!decoded.valid) { stats.failed_decodes++; continue; }

            random_resize_crop(decoded, resized_img.data(), TrainingConfig::IMAGE_SIZE, data_rng);

            std::uniform_real_distribution<T> flip_dist(0, 1);
            if (flip_dist(data_rng) < TrainingConfig::HFLIP_PROB) {
                constexpr TI S = TrainingConfig::IMAGE_SIZE;
                for (TI y = 0; y < S; y++) {
                    for (TI x = 0; x < S / 2; x++) {
                        for (TI c = 0; c < 3; c++) {
                            std::swap(resized_img[(y * S + x) * 3 + c],
                                      resized_img[(y * S + (S - 1 - x)) * 3 + c]);
                        }
                    }
                }
            }

            preprocess_to_tensor(device, resized_img.data(), input_tensor);
            rlt::forward(device, model, input_tensor, buffer, rng, train_mode);

            auto output_view = rlt::output(device, model);
            auto output_flat = rlt::view_memory<rlt::tensor::Shape<TI, TrainingConfig::NUM_CLASSES>>(device, output_view);

            T logits[TrainingConfig::NUM_CLASSES];
            for (TI i = 0; i < TrainingConfig::NUM_CLASSES; i++) {
                logits[i] = rlt::get(device, output_flat, i);
            }

            TI target = static_cast<TI>(s.label);
            batch_loss += cross_entropy_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING);

            TI predicted = 0;
            for (TI i = 1; i < TrainingConfig::NUM_CLASSES; i++) {
                if (logits[i] > logits[predicted]) predicted = i;
            }
            if (predicted == target) batch_correct++;
            batch_valid++;

            T d_logits[TrainingConfig::NUM_CLASSES];
            cross_entropy_gradient_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES,
                TrainingConfig::LABEL_SMOOTHING, T(1) / T(batch_size), d_logits);

            auto d_loss_flat = rlt::view_memory<rlt::tensor::Shape<TI, TrainingConfig::NUM_CLASSES>>(device, d_loss_tensor);
            for (TI i = 0; i < TrainingConfig::NUM_CLASSES; i++) {
                rlt::set(device, d_loss_flat, d_logits[i], i);
            }

            rlt::backward_full(device, model, input_tensor, d_loss_tensor, d_input_tensor, buffer);
        }

        rlt::step(device, optimizer, model);

        auto batch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> batch_duration = batch_end - batch_start;

        if (batch_valid > 0) batch_loss /= batch_valid;
        stats.loss_sum += batch_loss;
        stats.correct += batch_correct;
        stats.total += batch_valid;

        TI global_batch = batch_i_offset + batch_i;
        if (global_batch % log_interval == 0) {
            T batch_acc = batch_valid > 0 ? static_cast<T>(batch_correct) / batch_valid * 100.0f : 0;
            T samples_per_sec = batch_valid > 0 ? static_cast<T>(batch_valid) / batch_duration.count() : 0;
            std::cout << "  [" << global_batch << "/" << total_batches << "]"
                      << "  loss: " << batch_loss
                      << "  acc: " << batch_acc << "%"
                      << "  " << samples_per_sec << " samples/s"
                      << "  (" << batch_duration.count() << "s)"
                      << std::endl;
        }
    }
    return stats;
}

// ======================== Main ========================

int main(int argc, char* argv[]) {
    std::string dataset_dir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/git/imagenet-1k";
    std::string checkpoint_dir = "checkpoints";
    std::string resume_path = "";
    TI batch_size = 128;
    TI log_interval = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset-dir" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--checkpoint-dir" && i + 1 < argc) checkpoint_dir = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) log_interval = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --dataset-dir DIR    HuggingFace imagenet-1k directory (default: ~/git/imagenet-1k)\n"
                      << "  --checkpoint-dir DIR  Checkpoint output directory (default: checkpoints)\n"
                      << "  --resume PATH        Resume from HDF5 checkpoint\n"
                      << "  --batch-size N       Gradient accumulation batch size (default: 128)\n"
                      << "  --log-interval N     Log every N batches (default: 1)\n";
            return 0;
        }
    }

    // Discover parquet files
    auto train_files = find_parquet_files(dataset_dir, "train");
    auto val_files = find_parquet_files(dataset_dir, "validation");

    if (train_files.empty()) {
        std::cerr << "No training parquet files found in: " << dataset_dir << "/data/" << std::endl;
        return 1;
    }
    if (val_files.empty()) {
        std::cerr << "No validation parquet files found in: " << dataset_dir << "/data/" << std::endl;
        return 1;
    }

    // Count total training samples by reading metadata from first/last shards
    TI total_train_samples = 0;
    {
        for (auto& f : train_files) {
            auto maybe_file = arrow::io::ReadableFile::Open(f);
            if (!maybe_file.ok()) continue;
            auto maybe_reader = parquet::arrow::OpenFile(*maybe_file, arrow::default_memory_pool());
            if (!maybe_reader.ok()) continue;
            total_train_samples += (*maybe_reader)->parquet_reader()->metadata()->num_rows();
        }
    }
    TI total_batches = total_train_samples / batch_size;

    T scaled_lr = TrainingConfig::BASE_LR * static_cast<T>(batch_size) / static_cast<T>(TrainingConfig::BASE_BATCH_SIZE);

    std::cout << "=== ImageNet-1k ResNet-18 Training (timm recipe) ===" << std::endl;
    std::cout << "Dataset:         " << dataset_dir << std::endl;
    std::cout << "Train shards:    " << train_files.size() << " (" << total_train_samples << " samples)" << std::endl;
    std::cout << "Val shards:      " << val_files.size() << std::endl;
    std::cout << "Epochs:          " << TrainingConfig::NUM_EPOCHS << std::endl;
    std::cout << "Batch size:      " << batch_size << std::endl;
    std::cout << "Base LR:         " << TrainingConfig::BASE_LR << std::endl;
    std::cout << "Scaled LR:       " << scaled_lr << std::endl;
    std::cout << "Momentum:        " << TrainingConfig::MOMENTUM << std::endl;
    std::cout << "Weight decay:    " << TrainingConfig::WEIGHT_DECAY << std::endl;
    std::cout << "Nesterov:        " << (TrainingConfig::NESTEROV ? "true" : "false") << std::endl;
    std::cout << "Label smoothing: " << TrainingConfig::LABEL_SMOOTHING << std::endl;
    std::cout << "Warmup epochs:   " << TrainingConfig::WARMUP_EPOCHS << std::endl;

    // Initialize device, rng, model, optimizer
    DEVICE::SPEC::LOGGING logger;
    DEVICE device;
    device.logger = logger;

    std::mt19937 data_rng(42);
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 42);

    OPTIMIZER optimizer;
    RESNET18 model;
    typename RESNET18::template Buffer<true> buffer;

    rlt::malloc(device, optimizer);
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    rlt::init(device, optimizer);
    {
        auto opt_params = rlt::get(device, optimizer.parameters, 0);
        opt_params.learning_rate = scaled_lr;
        rlt::set(device, optimizer.parameters, opt_params, 0);
    }

    if (!resume_path.empty()) {
        std::cout << "Resuming from: " << resume_path << std::endl;
        auto file = HighFive::File(resume_path, HighFive::File::ReadOnly);
        auto model_group = rlt::get_group(device, file, "model");
        if (!rlt::load(device, model, model_group)) {
            std::cerr << "Failed to load model from checkpoint" << std::endl;
            return 1;
        }
        std::cout << "Model loaded from checkpoint" << std::endl;
    } else {
        rlt::init_weights(device, model, rng);
    }
    rlt::reset_optimizer_state(device, optimizer, model);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_tensor;
    rlt::malloc(device, input_tensor);

    using OUTPUT_SHAPE = typename RESNET18::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_loss_tensor;
    rlt::malloc(device, d_loss_tensor);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_tensor;
    rlt::malloc(device, d_input_tensor);

    std::vector<uint8_t> resized_img(TrainingConfig::IMAGE_SIZE * TrainingConfig::IMAGE_SIZE * 3);

    std::cout << "\nBatches per epoch: ~" << total_batches << std::endl << std::endl;

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr,
                                  TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS,
                                  TrainingConfig::WARMUP_LR);
        {
            auto opt_params = rlt::get(device, optimizer.parameters, 0);
            opt_params.learning_rate = current_lr;
            rlt::set(device, optimizer.parameters, opt_params, 0);
        }

        std::cout << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        // Shuffle shard order each epoch
        auto shuffled_train_files = train_files;
        std::shuffle(shuffled_train_files.begin(), shuffled_train_files.end(), data_rng);

        T epoch_loss = 0;
        TI epoch_correct = 0;
        TI epoch_total = 0;
        TI epoch_failed = 0;
        TI batch_offset = 0;

        for (TI shard_i = 0; shard_i < shuffled_train_files.size(); shard_i++) {
            ParquetShard shard;
            if (!shard.load(shuffled_train_files[shard_i])) {
                std::cerr << "Skipping shard: " << shuffled_train_files[shard_i] << std::endl;
                continue;
            }

            auto stats = train_on_shard(
                device, model, buffer, optimizer, rng, data_rng,
                shard, input_tensor, d_loss_tensor, d_input_tensor,
                resized_img, batch_offset, total_batches,
                batch_size, log_interval);

            epoch_loss += stats.loss_sum;
            epoch_correct += stats.correct;
            epoch_total += stats.total;
            epoch_failed += stats.failed_decodes;
            batch_offset += shard.samples.size() / batch_size;

            if ((shard_i + 1) % 50 == 0) {
                T running_acc = epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0;
                std::cout << "  Shards: " << (shard_i + 1) << "/" << shuffled_train_files.size()
                          << "  running acc: " << running_acc << "%" << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> epoch_duration = epoch_end - epoch_start;

        TI num_batches_done = batch_offset;
        T epoch_avg_loss = num_batches_done > 0 ? epoch_loss / num_batches_done : 0;
        T epoch_acc = epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0;

        std::cout << "  Train loss: " << epoch_avg_loss << std::endl;
        std::cout << "  Train acc:  " << epoch_acc << "%" << std::endl;
        std::cout << "  Time:       " << epoch_duration.count() << "s" << std::endl;
        if (epoch_failed > 0) std::cout << "  Failed decodes: " << epoch_failed << std::endl;

        // Validation
        {
            TI val_correct = 0, val_total = 0;
            T val_loss = 0;
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;

            for (auto& vf : val_files) {
                ParquetShard shard;
                if (!shard.load(vf)) continue;

                for (auto& s : shard.samples) {
                    auto decoded = decode_jpeg(s.image_bytes.data(), s.image_bytes.size());
                    if (!decoded.valid) continue;

                    center_crop_resize(decoded, resized_img.data(), TrainingConfig::IMAGE_SIZE);
                    preprocess_to_tensor(device, resized_img.data(), input_tensor);

                    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> val_output;
                    rlt::malloc(device, val_output);
                    rlt::evaluate(device, model, input_tensor, val_output, buffer, rng, eval_mode);

                    auto val_flat = rlt::view_memory<rlt::tensor::Shape<TI, TrainingConfig::NUM_CLASSES>>(device, val_output);
                    T logits[TrainingConfig::NUM_CLASSES];
                    for (TI j = 0; j < TrainingConfig::NUM_CLASSES; j++) {
                        logits[j] = rlt::get(device, val_flat, j);
                    }

                    TI target = static_cast<TI>(s.label);
                    val_loss += cross_entropy_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES, 0);

                    TI predicted = 0;
                    for (TI j = 1; j < TrainingConfig::NUM_CLASSES; j++) {
                        if (logits[j] > logits[predicted]) predicted = j;
                    }
                    if (predicted == target) val_correct++;
                    val_total++;

                    rlt::free(device, val_output);
                }
            }

            T val_avg_loss = val_total > 0 ? val_loss / val_total : 0;
            T val_acc = val_total > 0 ? static_cast<T>(val_correct) / val_total * 100.0f : 0;
            std::cout << "  Val loss:   " << val_avg_loss << std::endl;
            std::cout << "  Val acc:    " << val_acc << "% (top-1, " << val_total << " samples)" << std::endl;
        }

        // Checkpoint
        if ((epoch + 1) % TrainingConfig::CHECKPOINT_INTERVAL == 0 || epoch == TrainingConfig::NUM_EPOCHS - 1) {
            fs::create_directories(checkpoint_dir);
            std::string ckpt_path = checkpoint_dir + "/resnet18_epoch_" + std::to_string(epoch) + ".h5";
            auto file = HighFive::File(ckpt_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto model_group = rlt::create_group(device, file, "model");
            rlt::save(device, model, model_group);
            std::cout << "  Checkpoint: " << ckpt_path << std::endl;
        }
        std::cout << std::endl;
    }

    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, optimizer);
    rlt::free(device, input_tensor);
    rlt::free(device, d_loss_tensor);
    rlt::free(device, d_input_tensor);

    std::cout << "Training complete!" << std::endl;
    return 0;
}
