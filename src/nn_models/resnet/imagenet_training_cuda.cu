// ImageNet-1k training with ResNet-18, CUDA version
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>
#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>
#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/operations_cuda.h>
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
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <filesystem>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE_CPU = rlt::devices::DefaultCPU;
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

#ifndef MICRO_BATCH_SIZE
#define MICRO_BATCH_SIZE 64
#endif
constexpr TI_CUDA GPU_BATCH = MICRO_BATCH_SIZE;

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

struct SGDParams: rlt::nn::optimizers::sgd::DefaultParameters<TYPE_POLICY>{
    static constexpr T LEARNING_RATE = TrainingConfig::BASE_LR;
    static constexpr T MOMENTUM = TrainingConfig::MOMENTUM;
    static constexpr T WEIGHT_DECAY = TrainingConfig::WEIGHT_DECAY;
    static constexpr bool NESTEROV = TrainingConfig::NESTEROV;
    static constexpr bool ENABLE_WEIGHT_DECAY = true;
};

using OPTIMIZER_SPEC = rlt::nn::optimizers::sgd::Specification<TYPE_POLICY, TI_CUDA, SGDParams>;
using OPTIMIZER = rlt::nn::optimizers::SGD<OPTIMIZER_SPEC>;

using GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>;
using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, GPU_BATCH, TrainingConfig::IMAGE_SIZE, TrainingConfig::IMAGE_SIZE, 3>;
using RESNET18_CUDA = rlt::nn_models::sequential::Build<GPU_CAPABILITY, rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI_CUDA>, GPU_INPUT_SHAPE>;

using CPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>;
using CPU_INPUT_SHAPE = rlt::tensor::Shape<TI, GPU_BATCH, TrainingConfig::IMAGE_SIZE, TrainingConfig::IMAGE_SIZE, 3>;
using RESNET18_CPU = rlt::nn_models::sequential::Build<CPU_CAPABILITY, rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI>, CPU_INPUT_SHAPE>;

struct ParquetSample {
    std::vector<uint8_t> image_bytes;
    int64_t label;
};

struct ParquetShard {
    std::vector<ParquetSample> samples;
    bool load(const std::string& path) {
        auto maybe_file = arrow::io::ReadableFile::Open(path);
        if (!maybe_file.ok()) { std::cerr << "Failed to open: " << path << std::endl; return false; }
        auto maybe_reader = parquet::arrow::OpenFile(*maybe_file, arrow::default_memory_pool());
        if (!maybe_reader.ok()) { std::cerr << "Failed to open parquet: " << path << std::endl; return false; }
        auto& reader = *maybe_reader;
        std::shared_ptr<arrow::Table> table;
        auto status = reader->ReadTable(&table);
        if (!status.ok()) { std::cerr << "Failed to read table: " << path << std::endl; return false; }
        auto image_chunked = table->GetColumnByName("image");
        auto label_chunked = table->GetColumnByName("label");
        if (!image_chunked || !label_chunked) { std::cerr << "Missing columns in: " << path << std::endl; return false; }
        samples.clear();
        samples.reserve(table->num_rows());
        for (int ci = 0; ci < image_chunked->num_chunks(); ci++) {
            auto img_arr = std::static_pointer_cast<arrow::StructArray>(image_chunked->chunk(ci));
            auto lbl_arr = std::static_pointer_cast<arrow::Int64Array>(label_chunked->chunk(ci));
            auto bytes_field = img_arr->GetFieldByName("bytes");
            if (!bytes_field) { std::cerr << "Missing 'bytes' in: " << path << std::endl; return false; }
            auto bytes_arr = std::static_pointer_cast<arrow::BinaryArray>(bytes_field);
            for (int64_t row = 0; row < img_arr->length(); row++) {
                if (bytes_arr->IsNull(row) || lbl_arr->IsNull(row)) continue;
                auto view = bytes_arr->GetView(row);
                ParquetSample s;
                s.image_bytes.assign(reinterpret_cast<const uint8_t*>(view.data()),
                    reinterpret_cast<const uint8_t*>(view.data()) + view.size());
                s.label = lbl_arr->Value(row);
                samples.push_back(std::move(s));
            }
        }
        return true;
    }
};

std::vector<std::string> find_parquet_files(const std::string& base_dir, const std::string& split) {
    std::vector<std::string> files;
    std::string data_dir = base_dir + "/data";
    if (!fs::exists(data_dir)) data_dir = base_dir;
    for (auto& entry : fs::directory_iterator(data_dir)) {
        auto fname = entry.path().filename().string();
        if (fname.find(split + "-") == 0 && fname.find(".parquet") != std::string::npos
            && fname.find(".metadata") == std::string::npos)
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

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
    std::uniform_real_distribution<T> ratio_dist(std::log(TrainingConfig::CROP_RATIO_MIN), std::log(TrainingConfig::CROP_RATIO_MAX));
    int crop_w, crop_h, crop_x, crop_y;
    bool found = false;
    for (int attempt = 0; attempt < 10; attempt++) {
        T area = static_cast<T>(src.width * src.height);
        T target_area = scale_dist(rng) * area;
        T ratio = std::exp(ratio_dist(rng));
        crop_w = static_cast<int>(std::sqrt(target_area * ratio) + 0.5);
        crop_h = static_cast<int>(std::sqrt(target_area / ratio) + 0.5);
        if (crop_w > 0 && crop_w <= src.width && crop_h > 0 && crop_h <= src.height) {
            std::uniform_int_distribution<int> x_dist(0, src.width - crop_w);
            std::uniform_int_distribution<int> y_dist(0, src.height - crop_h);
            crop_x = x_dist(rng); crop_y = y_dist(rng);
            found = true; break;
        }
    }
    if (!found) {
        T img_ratio = static_cast<T>(src.width) / src.height;
        if (img_ratio < TrainingConfig::CROP_RATIO_MIN) { crop_w = src.width; crop_h = static_cast<int>(src.width / TrainingConfig::CROP_RATIO_MIN); }
        else if (img_ratio > TrainingConfig::CROP_RATIO_MAX) { crop_h = src.height; crop_w = static_cast<int>(src.height * TrainingConfig::CROP_RATIO_MAX); }
        else { crop_w = src.width; crop_h = src.height; }
        crop_w = std::min(crop_w, src.width); crop_h = std::min(crop_h, src.height);
        crop_x = (src.width - crop_w) / 2; crop_y = (src.height - crop_h) / 2;
    }
    std::vector<uint8_t> cropped(crop_w * crop_h * 3);
    for (int y = 0; y < crop_h; y++)
        std::memcpy(cropped.data() + y * crop_w * 3, src.pixels.data() + (crop_y + y) * src.width * 3 + crop_x * 3, crop_w * 3);
    stbir_resize_uint8_linear(cropped.data(), crop_w, crop_h, 0, dst, target_size, target_size, 0, STBIR_RGB);
}

void center_crop_resize(const DecodedImage& src, uint8_t* dst, TI target_size) {
    constexpr int RESIZE_SIZE = 256;
    T scale = static_cast<T>(RESIZE_SIZE) / std::min(src.width, src.height);
    int resized_w = std::max(static_cast<int>(src.width * scale + 0.5), RESIZE_SIZE);
    int resized_h = std::max(static_cast<int>(src.height * scale + 0.5), RESIZE_SIZE);
    std::vector<uint8_t> resized(resized_w * resized_h * 3);
    stbir_resize_uint8_linear(src.pixels.data(), src.width, src.height, 0, resized.data(), resized_w, resized_h, 0, STBIR_RGB);
    int x_off = (resized_w - static_cast<int>(target_size)) / 2;
    int y_off = (resized_h - static_cast<int>(target_size)) / 2;
    for (TI y = 0; y < target_size; y++)
        std::memcpy(dst + y * target_size * 3, resized.data() + (y_off + y) * resized_w * 3 + x_off * 3, target_size * 3);
}

void preprocess_to_buffer(const uint8_t* rgb_224, T* buffer, TI batch_offset) {
    constexpr TI SIZE = TrainingConfig::IMAGE_SIZE;
    constexpr TI IMG_ELEMS = SIZE * SIZE * 3;
    T* dst = buffer + batch_offset * IMG_ELEMS;
    for (TI h = 0; h < SIZE; h++)
        for (TI w = 0; w < SIZE; w++)
            for (TI c = 0; c < 3; c++) {
                T pixel = static_cast<T>(rgb_224[(h * SIZE + w) * 3 + c]) / 255.0f;
                dst[(h * SIZE + w) * 3 + c] = (pixel - static_cast<T>(rlt::nn_models::resnet18::IMAGENET_MEAN[c]))
                    / static_cast<T>(rlt::nn_models::resnet18::IMAGENET_STD[c]);
            }
}

T cosine_lr(TI epoch, TI total_epochs, T base_lr, T min_lr, TI warmup_epochs, T warmup_lr) {
    if (epoch < warmup_epochs)
        return warmup_lr + (base_lr - warmup_lr) * static_cast<T>(epoch) / static_cast<T>(warmup_epochs);
    T progress = static_cast<T>(epoch - warmup_epochs) / static_cast<T>(total_epochs - warmup_epochs);
    return min_lr + (base_lr - min_lr) * 0.5f * (1.0f + std::cos(static_cast<T>(M_PI) * progress));
}

T cross_entropy_with_smoothing(const T* logits, TI target, TI num_classes, T smoothing) {
    T max_logit = logits[0];
    for (TI i = 1; i < num_classes; i++) max_logit = std::max(max_logit, logits[i]);
    T sum_exp = 0;
    for (TI i = 0; i < num_classes; i++) sum_exp += std::exp(logits[i] - max_logit);
    T log_sum_exp = max_logit + std::log(sum_exp);
    T nll = -(logits[target] - log_sum_exp);
    T kl_uniform = log_sum_exp;
    for (TI i = 0; i < num_classes; i++) kl_uniform -= logits[i] / static_cast<T>(num_classes);
    return (1.0f - smoothing) * nll + smoothing * kl_uniform;
}

void cross_entropy_gradient_with_smoothing(const T* logits, TI target, TI num_classes, T smoothing, T loss_weight, T* d_logits) {
    T max_logit = logits[0];
    for (TI i = 1; i < num_classes; i++) max_logit = std::max(max_logit, logits[i]);
    T sum_exp = 0;
    for (TI i = 0; i < num_classes; i++) sum_exp += std::exp(logits[i] - max_logit);
    T smooth_weight = smoothing / static_cast<T>(num_classes);
    for (TI i = 0; i < num_classes; i++) {
        T softmax_i = std::exp(logits[i] - max_logit) / sum_exp;
        T smooth_target = (i == target) ? ((1.0f - smoothing) + smooth_weight) : smooth_weight;
        d_logits[i] = (softmax_i - smooth_target) * loss_weight;
    }
}

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
            std::cout << "Usage: " << argv[0] << " [--dataset-dir DIR] [--batch-size N] [--log-interval N] [--resume PATH]\n";
            return 0;
        }
    }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;
    auto train_files = find_parquet_files(dataset_dir, "train");
    auto val_files = find_parquet_files(dataset_dir, "validation");
    if (train_files.empty()) { std::cerr << "No training parquet files" << std::endl; return 1; }
    TI total_train_samples = 0;
    for (auto& f : train_files) {
        auto mf = arrow::io::ReadableFile::Open(f); if (!mf.ok()) continue;
        auto mr = parquet::arrow::OpenFile(*mf, arrow::default_memory_pool()); if (!mr.ok()) continue;
        total_train_samples += (*mr)->parquet_reader()->metadata()->num_rows();
    }
    T scaled_lr = TrainingConfig::BASE_LR * static_cast<T>(batch_size) / static_cast<T>(TrainingConfig::BASE_BATCH_SIZE);
    std::cout << "=== ImageNet ResNet-18 CUDA ===" << std::endl;
    std::cout << "Batch=" << batch_size << " micro=" << GPU_BATCH << " accum=" << num_micro_batches << " LR=" << scaled_lr << std::endl;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    std::mt19937 data_rng(42);
    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 42);

    OPTIMIZER optimizer;
    RESNET18_CUDA model;
    typename RESNET18_CUDA::template Buffer<true> model_buffer;
    rlt::malloc(device_cuda, optimizer); rlt::malloc(device_cuda, model); rlt::malloc(device_cuda, model_buffer);
    rlt::init(device_cuda, optimizer);
    { typename OPTIMIZER::PARAMETERS op; cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost); op.learning_rate = scaled_lr; cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice); }

    RESNET18_CPU model_cpu;
    rlt::malloc(device_cpu, model_cpu);
    if (!resume_path.empty()) {
        auto file = HighFive::File(resume_path, HighFive::File::ReadOnly);
        auto mg = rlt::get_group(device_cpu, file, "model");
        rlt::load(device_cpu, model_cpu, mg);
        rlt::copy(device_cpu, device_cuda, model_cpu, model);
    } else {
        DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 42);
        rlt::init_weights(device_cpu, model_cpu, rng_cpu);
        rlt::copy(device_cpu, device_cuda, model_cpu, model);
        rlt::free(device_cpu, rng_cpu);
    }
    rlt::reset_optimizer_state(device_cuda, optimizer, model);

    constexpr TI IMG_ELEMS = TrainingConfig::IMAGE_SIZE * TrainingConfig::IMAGE_SIZE * 3;
    constexpr TI OUTPUT_ELEMS = GPU_BATCH * TrainingConfig::NUM_CLASSES;
    using GPU_INPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input; rlt::malloc(device_cuda, gpu_input);
    using GPU_OUTPUT_SHAPE = typename RESNET18_CUDA::OUTPUT_SHAPE;
    using GPU_D_OUTPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_OUTPUT_SHAPE>;
    rlt::Tensor<GPU_D_OUTPUT_SPEC> gpu_d_output; rlt::malloc(device_cuda, gpu_d_output);
    rlt::Tensor<GPU_INPUT_SPEC> gpu_d_input; rlt::malloc(device_cuda, gpu_d_input);

    std::vector<T> cpu_input_buf(GPU_BATCH * IMG_ELEMS);
    std::vector<T> cpu_output_buf(OUTPUT_ELEMS);
    std::vector<T> cpu_d_output_buf(OUTPUT_ELEMS);
    std::vector<uint8_t> resized_img(IMG_ELEMS);
    std::vector<TI> batch_labels(GPU_BATCH);
    rlt::Mode<rlt::mode::Default<>> train_mode;
    TI total_batches = total_train_samples / batch_size;

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr, TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS, TrainingConfig::WARMUP_LR);
        { typename OPTIMIZER::PARAMETERS op; cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost); op.learning_rate = current_lr; cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice); }
        std::cout << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;
        auto shuffled_train_files = train_files;
        std::shuffle(shuffled_train_files.begin(), shuffled_train_files.end(), data_rng);
        T epoch_loss = 0; TI epoch_correct = 0, epoch_total = 0, epoch_failed = 0;
        std::vector<ParquetSample*> all_samples;
        std::vector<ParquetShard> shards(shuffled_train_files.size());
        for (TI si = 0; si < shuffled_train_files.size(); si++) {
            if (!shards[si].load(shuffled_train_files[si])) continue;
            for (auto& s : shards[si].samples) all_samples.push_back(&s);
        }
        std::shuffle(all_samples.begin(), all_samples.end(), data_rng);
        TI num_full_batches = all_samples.size() / batch_size;

        for (TI batch_i = 0; batch_i < num_full_batches; batch_i++) {
            auto batch_start = std::chrono::high_resolution_clock::now();
            rlt::zero_gradient(device_cuda, model);
            T batch_loss = 0; TI batch_correct = 0, batch_valid = 0;
            for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                TI base_idx = batch_i * batch_size + micro_i * GPU_BATCH;
                for (TI s_i = 0; s_i < GPU_BATCH; s_i++) {
                    auto* sample = all_samples[base_idx + s_i];
                    auto decoded = decode_jpeg(sample->image_bytes.data(), sample->image_bytes.size());
                    if (!decoded.valid) { epoch_failed++; std::memset(cpu_input_buf.data() + s_i * IMG_ELEMS, 0, IMG_ELEMS * sizeof(T)); batch_labels[s_i] = 0; continue; }
                    random_resize_crop(decoded, resized_img.data(), TrainingConfig::IMAGE_SIZE, data_rng);
                    std::uniform_real_distribution<T> flip_dist(0, 1);
                    if (flip_dist(data_rng) < TrainingConfig::HFLIP_PROB) {
                        constexpr TI S = TrainingConfig::IMAGE_SIZE;
                        for (TI y = 0; y < S; y++) for (TI x = 0; x < S / 2; x++) for (TI c = 0; c < 3; c++)
                            std::swap(resized_img[(y * S + x) * 3 + c], resized_img[(y * S + (S - 1 - x)) * 3 + c]);
                    }
                    preprocess_to_buffer(resized_img.data(), cpu_input_buf.data(), s_i);
                    batch_labels[s_i] = static_cast<TI>(sample->label);
                }
                cudaMemcpy(gpu_input._data, cpu_input_buf.data(), GPU_BATCH * IMG_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
                rlt::forward(device_cuda, model, gpu_input, model_buffer, rng_cuda, train_mode);
                auto output_view = rlt::output(device_cuda, model);
                cudaMemcpy(cpu_output_buf.data(), output_view._data, OUTPUT_ELEMS * sizeof(T), cudaMemcpyDeviceToHost);
                for (TI s_i = 0; s_i < GPU_BATCH; s_i++) {
                    const T* logits = cpu_output_buf.data() + s_i * TrainingConfig::NUM_CLASSES;
                    TI target = batch_labels[s_i];
                    batch_loss += cross_entropy_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING);
                    TI predicted = 0;
                    for (TI j = 1; j < TrainingConfig::NUM_CLASSES; j++) if (logits[j] > logits[predicted]) predicted = j;
                    if (predicted == target) batch_correct++;
                    batch_valid++;
                    cross_entropy_gradient_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES,
                        TrainingConfig::LABEL_SMOOTHING, T(1) / T(batch_size), cpu_d_output_buf.data() + s_i * TrainingConfig::NUM_CLASSES);
                }
                cudaMemcpy(gpu_d_output._data, cpu_d_output_buf.data(), OUTPUT_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
                rlt::backward_full(device_cuda, model, gpu_input, gpu_d_output, gpu_d_input, model_buffer);
            }
            rlt::step(device_cuda, optimizer, model);
            cudaDeviceSynchronize();
            auto batch_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<T> bd = batch_end - batch_start;
            if (batch_valid > 0) batch_loss /= batch_valid;
            epoch_loss += batch_loss; epoch_correct += batch_correct; epoch_total += batch_valid;
            if (batch_i % log_interval == 0) {
                T ba = batch_valid > 0 ? static_cast<T>(batch_correct) / batch_valid * 100.0f : 0;
                T sps = batch_valid > 0 ? static_cast<T>(batch_valid) / bd.count() : 0;
                std::cout << "  [" << batch_i << "/" << num_full_batches << "] loss=" << batch_loss << " acc=" << ba << "% " << sps << " img/s (" << bd.count() << "s)" << std::endl;
            }
        }
        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> ed = epoch_end - epoch_start;
        std::cout << "  Train loss=" << (num_full_batches > 0 ? epoch_loss / num_full_batches : 0) << " acc=" << (epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0) << "% time=" << ed.count() << "s" << std::endl;

        { // Validation
            TI vc = 0, vt = 0; T vl = 0;
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
            for (auto& vf : val_files) {
                ParquetShard shard; if (!shard.load(vf)) continue;
                TI nvb = shard.samples.size() / GPU_BATCH;
                for (TI vb = 0; vb < nvb; vb++) {
                    for (TI s_i = 0; s_i < GPU_BATCH; s_i++) {
                        auto& s = shard.samples[vb * GPU_BATCH + s_i];
                        auto d = decode_jpeg(s.image_bytes.data(), s.image_bytes.size());
                        if (!d.valid) { std::memset(cpu_input_buf.data() + s_i * IMG_ELEMS, 0, IMG_ELEMS * sizeof(T)); batch_labels[s_i] = 0; continue; }
                        center_crop_resize(d, resized_img.data(), TrainingConfig::IMAGE_SIZE);
                        preprocess_to_buffer(resized_img.data(), cpu_input_buf.data(), s_i);
                        batch_labels[s_i] = static_cast<TI>(s.label);
                    }
                    cudaMemcpy(gpu_input._data, cpu_input_buf.data(), GPU_BATCH * IMG_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
                    rlt::Tensor<GPU_D_OUTPUT_SPEC> vo; rlt::malloc(device_cuda, vo);
                    rlt::evaluate(device_cuda, model, gpu_input, vo, model_buffer, rng_cuda, eval_mode);
                    cudaMemcpy(cpu_output_buf.data(), vo._data, OUTPUT_ELEMS * sizeof(T), cudaMemcpyDeviceToHost);
                    rlt::free(device_cuda, vo);
                    for (TI s_i = 0; s_i < GPU_BATCH; s_i++) {
                        const T* logits = cpu_output_buf.data() + s_i * TrainingConfig::NUM_CLASSES;
                        TI target = batch_labels[s_i];
                        vl += cross_entropy_with_smoothing(logits, target, TrainingConfig::NUM_CLASSES, 0);
                        TI p = 0; for (TI j = 1; j < TrainingConfig::NUM_CLASSES; j++) if (logits[j] > logits[p]) p = j;
                        if (p == target) vc++; vt++;
                    }
                }
            }
            std::cout << "  Val loss=" << (vt > 0 ? vl / vt : 0) << " acc=" << (vt > 0 ? static_cast<T>(vc) / vt * 100.0f : 0) << "%" << std::endl;
        }

        if ((epoch + 1) % TrainingConfig::CHECKPOINT_INTERVAL == 0 || epoch == TrainingConfig::NUM_EPOCHS - 1) {
            rlt::copy(device_cuda, device_cpu, model, model_cpu);
            fs::create_directories(checkpoint_dir);
            std::string ckpt = checkpoint_dir + "/resnet18_cuda_epoch_" + std::to_string(epoch) + ".h5";
            auto file = HighFive::File(ckpt, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto mg = rlt::create_group(device_cpu, file, "model");
            rlt::save(device_cpu, model_cpu, mg);
            std::cout << "  Checkpoint: " << ckpt << std::endl;
        }
        std::cout << std::endl;
    }
    rlt::free(device_cuda, model); rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, optimizer); rlt::free(device_cuda, gpu_input);
    rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_input);
    rlt::free(device_cuda, rng_cuda); rlt::free(device_cpu, model_cpu);
    std::cout << "Training complete!" << std::endl;
    return 0;
}
