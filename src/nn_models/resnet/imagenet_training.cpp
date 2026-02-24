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
// Data format: binary file produced by prepare_imagenet.py
//   Header:  [num_images: u64]
//   Index:   [offset: u64, size: u32, label: u32] * num_images
//   Data:    concatenated JPEG bytes

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

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE = rlt::devices::DEVICE_FACTORY<rlt::devices::DefaultCPUSpecification>;
using TI = typename DEVICE::index_t;

// ======================== Hyperparameters (timm resnet18.a1_in1k recipe) ========================

struct TrainingConfig {
    static constexpr TI IMAGE_SIZE = 224;
    static constexpr TI NUM_CLASSES = 1000;
    static constexpr TI BATCH_SIZE = 128;
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
    static constexpr T CROP_RATIO_MIN = 0.75;  // 3/4
    static constexpr T CROP_RATIO_MAX = 1.3333; // 4/3
    static constexpr T HFLIP_PROB = 0.5;
    static constexpr TI LOG_INTERVAL = 100;
    static constexpr TI CHECKPOINT_INTERVAL = 10;
    static constexpr TI VAL_SAMPLES = 50000;
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

// ======================== Binary dataset reader ========================

struct IndexEntry {
    uint64_t offset;
    uint32_t size;
    uint32_t label;
};

struct Dataset {
    std::vector<uint8_t> data;
    std::vector<IndexEntry> index;
    uint64_t num_images = 0;

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) {
            std::cerr << "Failed to open dataset: " << path << std::endl;
            return false;
        }
        f.seekg(0, std::ios::end);
        size_t file_size = f.tellg();
        f.seekg(0);

        f.read(reinterpret_cast<char*>(&num_images), sizeof(uint64_t));
        index.resize(num_images);
        f.read(reinterpret_cast<char*>(index.data()), num_images * sizeof(IndexEntry));

        size_t header_size = sizeof(uint64_t) + num_images * sizeof(IndexEntry);
        size_t data_size = file_size - header_size;
        data.resize(data_size);
        f.read(reinterpret_cast<char*>(data.data()), data_size);

        std::cout << "Loaded dataset: " << path << " (" << num_images << " images, "
                  << (file_size / (1024.0 * 1024.0 * 1024.0)) << " GB)" << std::endl;
        return true;
    }

    const uint8_t* jpeg_data(TI idx) const {
        size_t header_size = sizeof(uint64_t) + num_images * sizeof(IndexEntry);
        return data.data() + (index[idx].offset - header_size);
    }

    uint32_t jpeg_size(TI idx) const {
        return index[idx].size;
    }

    uint32_t label(TI idx) const {
        return index[idx].label;
    }
};

// ======================== Image preprocessing ========================

struct DecodedImage {
    std::vector<uint8_t> pixels;
    int width = 0, height = 0;
    bool valid = false;
};

DecodedImage decode_jpeg(const uint8_t* jpeg_bytes, uint32_t jpeg_size) {
    DecodedImage img;
    int channels;
    uint8_t* raw = stbi_load_from_memory(jpeg_bytes, jpeg_size, &img.width, &img.height, &channels, 3);
    if (!raw) {
        return img;
    }
    img.pixels.assign(raw, raw + img.width * img.height * 3);
    stbi_image_free(raw);
    img.valid = true;
    return img;
}

// Random resize crop: pick random area and aspect ratio, crop, then resize to target_size
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

// Center crop for validation: resize shorter side to 256, then center crop to target_size
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

T cross_entropy_with_smoothing(DEVICE& device, const T* logits, TI target, TI num_classes, T smoothing) {
    T max_logit = logits[0];
    for (TI i = 1; i < num_classes; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }
    T sum_exp = 0;
    for (TI i = 0; i < num_classes; i++) {
        sum_exp += std::exp(logits[i] - max_logit);
    }
    T log_sum_exp = max_logit + std::log(sum_exp);

    T target_weight = 1.0f - smoothing;
    T smooth_weight = smoothing / static_cast<T>(num_classes);

    T nll = -(logits[target] - log_sum_exp);
    T kl_uniform = log_sum_exp;
    for (TI i = 0; i < num_classes; i++) {
        kl_uniform -= logits[i] / static_cast<T>(num_classes);
    }

    return target_weight * nll + smoothing * kl_uniform;
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

    T target_weight = 1.0f - smoothing;
    T smooth_weight = smoothing / static_cast<T>(num_classes);

    for (TI i = 0; i < num_classes; i++) {
        T softmax_i = std::exp(logits[i] - max_logit) / sum_exp;
        T smooth_target = (i == target) ? (target_weight + smooth_weight) : smooth_weight;
        d_logits[i] = (softmax_i - smooth_target) * loss_weight;
    }
}

// ======================== Main ========================

int main(int argc, char* argv[]) {
    std::string train_data_path = "imagenet_bin/train.bin";
    std::string val_data_path = "imagenet_bin/val.bin";
    std::string checkpoint_dir = "checkpoints";
    std::string resume_path = "";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--train-data" && i + 1 < argc) train_data_path = argv[++i];
        else if (arg == "--val-data" && i + 1 < argc) val_data_path = argv[++i];
        else if (arg == "--checkpoint-dir" && i + 1 < argc) checkpoint_dir = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  --train-data PATH    Training binary dataset (default: imagenet_bin/train.bin)\n"
                      << "  --val-data PATH      Validation binary dataset (default: imagenet_bin/val.bin)\n"
                      << "  --checkpoint-dir DIR  Checkpoint output directory (default: checkpoints)\n"
                      << "  --resume PATH        Resume from HDF5 checkpoint\n";
            return 0;
        }
    }

    std::cout << "=== ImageNet-1k ResNet-18 Training (timm recipe) ===" << std::endl;
    std::cout << "Epochs:          " << TrainingConfig::NUM_EPOCHS << std::endl;
    std::cout << "Batch size:      " << TrainingConfig::BATCH_SIZE << std::endl;
    std::cout << "Base LR:         " << TrainingConfig::BASE_LR << std::endl;
    std::cout << "Momentum:        " << TrainingConfig::MOMENTUM << std::endl;
    std::cout << "Weight decay:    " << TrainingConfig::WEIGHT_DECAY << std::endl;
    std::cout << "Nesterov:        " << (TrainingConfig::NESTEROV ? "true" : "false") << std::endl;
    std::cout << "Label smoothing: " << TrainingConfig::LABEL_SMOOTHING << std::endl;
    std::cout << "Warmup epochs:   " << TrainingConfig::WARMUP_EPOCHS << std::endl;

    // Load datasets
    Dataset train_dataset, val_dataset;
    if (!train_dataset.load(train_data_path)) return 1;
    if (!val_dataset.load(val_data_path)) return 1;

    // Scale learning rate by effective batch size
    T scaled_lr = TrainingConfig::BASE_LR * static_cast<T>(TrainingConfig::BATCH_SIZE) / static_cast<T>(TrainingConfig::BASE_BATCH_SIZE);
    std::cout << "Scaled LR:       " << scaled_lr << " (batch " << TrainingConfig::BATCH_SIZE << " / base " << TrainingConfig::BASE_BATCH_SIZE << ")" << std::endl;

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

    TI start_epoch = 0;
    if (!resume_path.empty()) {
        std::cout << "Resuming from: " << resume_path << std::endl;
        auto file = HighFive::File(resume_path, HighFive::File::ReadOnly);
        auto model_group = rlt::get_group(device, file, "model");
        bool load_ok = rlt::load(device, model, model_group);
        if (!load_ok) {
            std::cerr << "Failed to load model from checkpoint" << std::endl;
            return 1;
        }
        std::cout << "Model loaded from checkpoint" << std::endl;
    } else {
        rlt::init_weights(device, model, rng);
    }
    rlt::reset_optimizer_state(device, optimizer, model);

    // Allocate input/output tensors for single-sample forward/backward
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input_tensor;
    rlt::malloc(device, input_tensor);

    using OUTPUT_SHAPE = typename RESNET18::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> d_loss_tensor;
    rlt::malloc(device, d_loss_tensor);

    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> d_input_tensor;
    rlt::malloc(device, d_input_tensor);

    // Temp buffers for image preprocessing
    std::vector<uint8_t> resized_img(TrainingConfig::IMAGE_SIZE * TrainingConfig::IMAGE_SIZE * 3);

    // Shuffle indices
    std::vector<TI> train_indices(train_dataset.num_images);
    std::iota(train_indices.begin(), train_indices.end(), 0);

    TI num_batches = train_dataset.num_images / TrainingConfig::BATCH_SIZE;

    std::cout << "\nTraining: " << train_dataset.num_images << " images, " << num_batches << " batches/epoch" << std::endl;
    std::cout << "Validation: " << val_dataset.num_images << " images" << std::endl;
    std::cout << std::endl;

    for (TI epoch = start_epoch; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();

        // Update learning rate
        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr,
                                  TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS,
                                  TrainingConfig::WARMUP_LR);
        {
            auto opt_params = rlt::get(device, optimizer.parameters, 0);
            opt_params.learning_rate = current_lr;
            rlt::set(device, optimizer.parameters, opt_params, 0);
        }

        // Shuffle training data
        std::shuffle(train_indices.begin(), train_indices.end(), data_rng);

        T epoch_loss = 0;
        TI epoch_correct = 0;
        TI epoch_total = 0;
        TI failed_decodes = 0;
        rlt::Mode<rlt::mode::Default<>> train_mode;

        for (TI batch_i = 0; batch_i < num_batches; batch_i++) {
            rlt::zero_gradient(device, model);
            T batch_loss = 0;
            TI batch_correct = 0;
            TI batch_valid = 0;

            for (TI sample_i = 0; sample_i < TrainingConfig::BATCH_SIZE; sample_i++) {
                TI global_idx = train_indices[batch_i * TrainingConfig::BATCH_SIZE + sample_i];
                auto decoded = decode_jpeg(train_dataset.jpeg_data(global_idx), train_dataset.jpeg_size(global_idx));
                if (!decoded.valid) {
                    failed_decodes++;
                    continue;
                }

                // Random resize crop + horizontal flip
                random_resize_crop(decoded, resized_img.data(), TrainingConfig::IMAGE_SIZE, data_rng);

                std::uniform_real_distribution<T> flip_dist(0, 1);
                if (flip_dist(data_rng) < TrainingConfig::HFLIP_PROB) {
                    constexpr TI S = TrainingConfig::IMAGE_SIZE;
                    for (TI y = 0; y < S; y++) {
                        for (TI x = 0; x < S / 2; x++) {
                            for (TI c = 0; c < 3; c++) {
                                std::swap(
                                    resized_img[(y * S + x) * 3 + c],
                                    resized_img[(y * S + (S - 1 - x)) * 3 + c]);
                            }
                        }
                    }
                }

                preprocess_to_tensor(device, resized_img.data(), input_tensor);

                rlt::forward(device, model, input_tensor, buffer, rng, train_mode);

                // Get output logits
                auto output_view = rlt::output(device, model);
                auto output_flat = rlt::view_memory<rlt::tensor::Shape<TI, TrainingConfig::NUM_CLASSES>>(device, output_view);

                T logits[TrainingConfig::NUM_CLASSES];
                for (TI i = 0; i < TrainingConfig::NUM_CLASSES; i++) {
                    logits[i] = rlt::get(device, output_flat, i);
                }

                TI target = train_dataset.label(global_idx);
                T sample_loss = cross_entropy_with_smoothing(device, logits, target, TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING);
                batch_loss += sample_loss;

                // Top-1 accuracy
                TI predicted = 0;
                for (TI i = 1; i < TrainingConfig::NUM_CLASSES; i++) {
                    if (logits[i] > logits[predicted]) predicted = i;
                }
                if (predicted == target) batch_correct++;
                batch_valid++;

                // Compute gradient of loss w.r.t. output
                T d_logits[TrainingConfig::NUM_CLASSES];
                cross_entropy_gradient_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES,
                    TrainingConfig::LABEL_SMOOTHING, T(1) / T(TrainingConfig::BATCH_SIZE), d_logits);

                auto d_loss_flat = rlt::view_memory<rlt::tensor::Shape<TI, TrainingConfig::NUM_CLASSES>>(device, d_loss_tensor);
                for (TI i = 0; i < TrainingConfig::NUM_CLASSES; i++) {
                    rlt::set(device, d_loss_flat, d_logits[i], i);
                }

                rlt::backward_full(device, model, input_tensor, d_loss_tensor, d_input_tensor, buffer);
            }

            rlt::step(device, optimizer, model);

            if (batch_valid > 0) {
                batch_loss /= batch_valid;
            }
            epoch_loss += batch_loss;
            epoch_correct += batch_correct;
            epoch_total += batch_valid;

            if (batch_i % TrainingConfig::LOG_INTERVAL == 0) {
                T batch_acc = batch_valid > 0 ? static_cast<T>(batch_correct) / batch_valid * 100.0f : 0;
                std::cout << "Epoch " << epoch << " [" << batch_i << "/" << num_batches << "]"
                          << "  loss: " << batch_loss
                          << "  acc: " << batch_acc << "%"
                          << "  lr: " << current_lr
                          << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> epoch_duration = epoch_end - epoch_start;

        T epoch_avg_loss = num_batches > 0 ? epoch_loss / num_batches : 0;
        T epoch_acc = epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0;

        std::cout << "\n=== Epoch " << epoch << " Summary ===" << std::endl;
        std::cout << "  Train loss: " << epoch_avg_loss << std::endl;
        std::cout << "  Train acc:  " << epoch_acc << "%" << std::endl;
        std::cout << "  Time:       " << epoch_duration.count() << "s" << std::endl;
        std::cout << "  LR:         " << current_lr << std::endl;
        if (failed_decodes > 0) {
            std::cout << "  Failed decodes: " << failed_decodes << std::endl;
        }

        // Validation
        {
            TI val_correct = 0;
            TI val_total = 0;
            T val_loss = 0;
            TI val_count = std::min(static_cast<TI>(val_dataset.num_images), TrainingConfig::VAL_SAMPLES);
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;

            for (TI i = 0; i < val_count; i++) {
                auto decoded = decode_jpeg(val_dataset.jpeg_data(i), val_dataset.jpeg_size(i));
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

                TI target = val_dataset.label(i);
                val_loss += cross_entropy_with_smoothing(device, logits, target, TrainingConfig::NUM_CLASSES, 0);

                TI predicted = 0;
                for (TI j = 1; j < TrainingConfig::NUM_CLASSES; j++) {
                    if (logits[j] > logits[predicted]) predicted = j;
                }
                if (predicted == target) val_correct++;
                val_total++;

                rlt::free(device, val_output);

                if ((i + 1) % 10000 == 0) {
                    std::cout << "  Validation progress: " << (i + 1) << "/" << val_count << std::endl;
                }
            }

            T val_avg_loss = val_total > 0 ? val_loss / val_total : 0;
            T val_acc = val_total > 0 ? static_cast<T>(val_correct) / val_total * 100.0f : 0;
            std::cout << "  Val loss:   " << val_avg_loss << std::endl;
            std::cout << "  Val acc:    " << val_acc << "% (top-1)" << std::endl;
        }

        // Checkpoint
        if ((epoch + 1) % TrainingConfig::CHECKPOINT_INTERVAL == 0 || epoch == TrainingConfig::NUM_EPOCHS - 1) {
            std::string ckpt_path = checkpoint_dir + "/resnet18_epoch_" + std::to_string(epoch) + ".h5";
            {
                std::string mkdir_cmd = "mkdir -p " + checkpoint_dir;
                int ret = system(mkdir_cmd.c_str());
                if (ret != 0) {
                    std::cerr << "Warning: mkdir failed for " << checkpoint_dir << std::endl;
                }
            }
            auto file = HighFive::File(ckpt_path, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto model_group = rlt::create_group(device, file, "model");
            rlt::save(device, model, model_group);
            std::cout << "  Checkpoint: " << ckpt_path << std::endl;
        }
        std::cout << std::endl;
    }

    // Cleanup
    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, optimizer);
    rlt::free(device, input_tensor);
    rlt::free(device, d_loss_tensor);
    rlt::free(device, d_input_tensor);

    std::cout << "Training complete!" << std::endl;
    return 0;
}
