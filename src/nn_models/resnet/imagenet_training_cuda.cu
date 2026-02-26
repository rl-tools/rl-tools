#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
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

#include "image_preprocess.h"

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <thread>
#include <atomic>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

#ifndef MICRO_BATCH_SIZE
#define MICRO_BATCH_SIZE 64
#endif
constexpr TI_CUDA GPU_BATCH = MICRO_BATCH_SIZE;

// --- Training hyperparameters ---
struct TrainingConfig {
    static constexpr TI IMAGE_SIZE = 224;
    static constexpr TI NUM_CLASSES = 1000;
    static constexpr TI NUM_EPOCHS = 90;
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
    static constexpr TI CHECKPOINT_INTERVAL = 1;
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

// --- GPU cross-entropy loss + gradient kernel ---
__global__ void cross_entropy_loss_gradient_kernel(
    const T* __restrict__ logits,    // [N, C]
    const TI_CUDA* __restrict__ labels, // [N]
    T* __restrict__ d_logits,        // [N, C]
    T* __restrict__ losses,          // [N]
    TI_CUDA* __restrict__ correct,   // [N] (1 if top-1 correct)
    TI_CUDA* __restrict__ correct5,  // [N] (1 if top-5 correct)
    TI_CUDA num_classes,
    T smoothing,
    T loss_weight
) {
    TI_CUDA i = blockIdx.x;
    const T* row = logits + i * num_classes;
    T* d_row = d_logits + i * num_classes;
    TI_CUDA target = labels[i];

    T max_logit = row[0];
    for (TI_CUDA c = 1; c < num_classes; c++) max_logit = max(max_logit, row[c]);
    T sum_exp = 0;
    for (TI_CUDA c = 0; c < num_classes; c++) sum_exp += expf(row[c] - max_logit);
    T log_sum_exp = max_logit + logf(sum_exp);

    T nll = -(row[target] - log_sum_exp);
    T kl_uniform = log_sum_exp;
    for (TI_CUDA c = 0; c < num_classes; c++) kl_uniform -= row[c] / static_cast<T>(num_classes);
    losses[i] = (1.0f - smoothing) * nll + smoothing * kl_uniform;

    T smooth_weight = smoothing / static_cast<T>(num_classes);
    for (TI_CUDA c = 0; c < num_classes; c++) {
        T softmax_c = expf(row[c] - max_logit) / sum_exp;
        T smooth_target = (c == target) ? ((1.0f - smoothing) + smooth_weight) : smooth_weight;
        d_row[c] = (softmax_c - smooth_target) * loss_weight;
    }

    // Top-1
    TI_CUDA predicted = 0;
    for (TI_CUDA c = 1; c < num_classes; c++) if (row[c] > row[predicted]) predicted = c;
    correct[i] = (predicted == target) ? 1 : 0;

    // Top-5
    TI_CUDA count = 0;
    T target_logit = row[target];
    for (TI_CUDA c = 0; c < num_classes; c++) {
        if (row[c] > target_logit) count++;
        if (count >= 5) break;
    }
    correct5[i] = (count < 5) ? 1 : 0;
}

// --- Parquet data loading ---
struct ParquetSample {
    const uint8_t* image_data;
    size_t image_size;
    int64_t label;
};

struct ParquetShard {
    std::shared_ptr<arrow::Table> table_;
    std::vector<ParquetSample> samples;
    bool load(const std::string& path) {
        auto maybe_file = arrow::io::ReadableFile::Open(path);
        if (!maybe_file.ok()) { std::cerr << "Failed to open: " << path << std::endl; return false; }
        auto maybe_reader = parquet::arrow::OpenFile(*maybe_file, arrow::default_memory_pool());
        if (!maybe_reader.ok()) { std::cerr << "Failed to open parquet: " << path << std::endl; return false; }
        auto status = (*maybe_reader)->ReadTable(&table_);
        if (!status.ok()) { std::cerr << "Failed to read table: " << path << std::endl; return false; }
        auto image_chunked = table_->GetColumnByName("image");
        auto label_chunked = table_->GetColumnByName("label");
        if (!image_chunked || !label_chunked) { std::cerr << "Missing columns in: " << path << std::endl; return false; }
        samples.clear();
        samples.reserve(table_->num_rows());
        for (int ci = 0; ci < image_chunked->num_chunks(); ci++) {
            auto img_arr = std::static_pointer_cast<arrow::StructArray>(image_chunked->chunk(ci));
            auto lbl_arr = std::static_pointer_cast<arrow::Int64Array>(label_chunked->chunk(ci));
            auto bytes_field = img_arr->GetFieldByName("bytes");
            if (!bytes_field) { std::cerr << "Missing 'bytes' in: " << path << std::endl; return false; }
            auto bytes_arr = std::static_pointer_cast<arrow::BinaryArray>(bytes_field);
            for (int64_t row = 0; row < img_arr->length(); row++) {
                if (bytes_arr->IsNull(row) || lbl_arr->IsNull(row)) continue;
                auto view = bytes_arr->GetView(row);
                samples.push_back({reinterpret_cast<const uint8_t*>(view.data()), static_cast<size_t>(view.size()), lbl_arr->Value(row)});
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
        if (fname.find(split + "-") == 0 && fname.find(".parquet") != std::string::npos && fname.find(".metadata") == std::string::npos)
            files.push_back(entry.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

// --- Preprocessing ---
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

int main(int argc, char* argv[]) {
    std::string dataset_dir = std::string(getenv("HOME") ? getenv("HOME") : ".") + "/git/imagenet-1k";
    auto now_tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ts_ss;
    ts_ss << std::put_time(std::localtime(&now_tt), "%Y-%m-%dT%H-%M-%S");
    std::string logdir = "runs/" + ts_ss.str();
    std::string resume_path = "";
    TI batch_size = 256;
    TI log_interval = 1;
    bool preload = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--dataset-dir" && i + 1 < argc) dataset_dir = argv[++i];
        else if (arg == "--logdir" && i + 1 < argc) logdir = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) log_interval = std::stoi(argv[++i]);
        else if (arg == "--preload") preload = true;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [--dataset-dir DIR] [--batch-size N] [--log-interval N] [--logdir DIR] [--resume PATH] [--preload]\n";
            return 0;
        }
    }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;

    std::cout << "Dataset dir: " << dataset_dir << std::endl;
    std::cout << "Scanning parquet files..." << std::flush;
    auto train_files = find_parquet_files(dataset_dir, "train");
    auto val_files = find_parquet_files(dataset_dir, "validation");
    std::cout << " found " << train_files.size() << " train, " << val_files.size() << " val" << std::endl;
    if (train_files.empty()) { std::cerr << "No training parquet files in " << dataset_dir << std::endl; return 1; }

    std::cout << "Counting samples..." << std::flush;
    TI total_train_samples = 0;
    for (auto& f : train_files) {
        auto mf = arrow::io::ReadableFile::Open(f); if (!mf.ok()) continue;
        auto mr = parquet::arrow::OpenFile(*mf, arrow::default_memory_pool()); if (!mr.ok()) continue;
        total_train_samples += (*mr)->parquet_reader()->metadata()->num_rows();
    }
    std::cout << " " << total_train_samples << " training images" << std::endl;

    std::vector<ParquetShard> preloaded_train_shards;
    std::vector<ParquetSample> preloaded_train_samples;
    std::vector<ParquetShard> preloaded_val_shards;
    std::vector<ParquetSample> preloaded_val_samples;
    if (preload) {
        std::cout << "Preloading training data..." << std::flush;
        preloaded_train_shards.reserve(train_files.size());
        for (auto& f : train_files) {
            preloaded_train_shards.emplace_back();
            if (preloaded_train_shards.back().load(f)) {
                for (auto& s : preloaded_train_shards.back().samples)
                    preloaded_train_samples.push_back(s);
            } else {
                preloaded_train_shards.pop_back();
            }
        }
        std::cout << " " << preloaded_train_samples.size() << " samples from " << preloaded_train_shards.size() << " shards" << std::endl;
        std::cout << "Preloading validation data..." << std::flush;
        preloaded_val_shards.reserve(val_files.size());
        for (auto& f : val_files) {
            preloaded_val_shards.emplace_back();
            if (preloaded_val_shards.back().load(f)) {
                for (auto& s : preloaded_val_shards.back().samples)
                    preloaded_val_samples.push_back(s);
            } else {
                preloaded_val_shards.pop_back();
            }
        }
        std::cout << " " << preloaded_val_samples.size() << " samples" << std::endl;
        total_train_samples = preloaded_train_samples.size();
    }

    T scaled_lr = TrainingConfig::BASE_LR * static_cast<T>(batch_size) / static_cast<T>(TrainingConfig::BASE_BATCH_SIZE);

    std::cout << "=== ImageNet ResNet-18 CUDA ===" << std::endl;
    std::cout << "Batch=" << batch_size << " micro=" << GPU_BATCH << " accum=" << num_micro_batches << " LR=" << scaled_lr << std::endl;

    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    rlt::init(device_cpu, device_cpu.logger, fs::path(logdir));
    std::cout << "TensorBoard logdir: " << logdir << std::endl;

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

    // GPU buffers for loss computation (no CPU round-trip needed)
    TI_CUDA* gpu_labels; cudaMalloc(&gpu_labels, GPU_BATCH * sizeof(TI_CUDA));
    T* gpu_losses; cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(T));
    TI_CUDA* gpu_correct; cudaMalloc(&gpu_correct, GPU_BATCH * sizeof(TI_CUDA));
    TI_CUDA* gpu_correct5; cudaMalloc(&gpu_correct5, GPU_BATCH * sizeof(TI_CUDA));

    // Regular staging buffer for multi-threaded decode (pinned memory is slow with concurrent writes)
    std::vector<T> cpu_input_staging(GPU_BATCH * IMG_ELEMS);
    // Pinned host buffer for fast H2D transfer
    T* cpu_input_pinned; cudaHostAlloc(&cpu_input_pinned, GPU_BATCH * IMG_ELEMS * sizeof(T), cudaHostAllocDefault);
    std::vector<TI_CUDA> cpu_labels(GPU_BATCH);

    TI num_workers = std::min(static_cast<TI>(std::thread::hardware_concurrency()), static_cast<TI>(GPU_BATCH));
    if (num_workers < 1) num_workers = 8;
    std::cout << "Data workers: " << num_workers << std::endl;

    std::vector<std::vector<uint8_t>> worker_resize_bufs(GPU_BATCH, std::vector<uint8_t>(IMG_ELEMS));
    std::vector<std::vector<uint8_t>> val_scratch_bufs(GPU_BATCH);

    rlt::Mode<rlt::mode::Default<>> train_mode;
    TI total_batches = total_train_samples / batch_size;
    TI global_step = 0;

    { // Decode benchmark
        ParquetShard bench_shard;
        if (bench_shard.load(train_files[0])) {
            constexpr TI BENCH_N = 64;
            std::vector<uint8_t> resize_buf(IMG_ELEMS);
            std::mt19937 bench_rng(0);
            // Warmup
            for (TI i = 0; i < 4; i++) {
                auto d = decode_jpeg(bench_shard.samples[i].image_data, bench_shard.samples[i].image_size);
                random_resize_crop(d, resize_buf.data(), TrainingConfig::IMAGE_SIZE, 0.08f, 1.0f, 0.75f, 1.333f, bench_rng);
            }
            // Single-threaded benchmark
            auto t0 = std::chrono::high_resolution_clock::now();
            for (TI i = 0; i < BENCH_N; i++) {
                auto d = decode_jpeg(bench_shard.samples[i].image_data, bench_shard.samples[i].image_size);
                random_resize_crop(d, resize_buf.data(), TrainingConfig::IMAGE_SIZE, 0.08f, 1.0f, 0.75f, 1.333f, bench_rng);
                preprocess_to_buffer(resize_buf.data(), cpu_input_staging.data(), 0);
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            double st_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cout << "Bench: 1 thread, " << BENCH_N << " images: " << st_ms << "ms (" << st_ms / BENCH_N << " ms/img, " << BENCH_N * 1000.0 / st_ms << " img/s)" << std::endl;
            for (TI nw : {4u, 8u, 16u, 20u}) {
                auto t2 = std::chrono::high_resolution_clock::now();
                constexpr TI MT_N = 256;
                std::vector<std::thread> threads;
                TI per_w = (MT_N + nw - 1) / nw;
                for (TI w = 0; w < nw; w++) {
                    TI s = w * per_w, e = std::min(s + per_w, MT_N);
                    if (s >= MT_N) break;
                    threads.emplace_back([&, s, e]() {
                        std::vector<uint8_t> rbuf(IMG_ELEMS);
                        std::mt19937 rng(s);
                        for (TI i = s; i < e; i++) {
                            auto d = decode_jpeg(bench_shard.samples[i].image_data, bench_shard.samples[i].image_size);
                            random_resize_crop(d, rbuf.data(), TrainingConfig::IMAGE_SIZE, 0.08f, 1.0f, 0.75f, 1.333f, rng);
                            preprocess_to_buffer(rbuf.data(), cpu_input_staging.data(), i);
                        }
                    });
                }
                for (auto& t : threads) t.join();
                auto t3 = std::chrono::high_resolution_clock::now();
                double mt_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
                std::cout << "Bench: " << nw << " threads, " << MT_N << " images: " << mt_ms << "ms (" << mt_ms / MT_N << " ms/img, " << MT_N * 1000.0 / mt_ms << " img/s)" << std::endl;
            }
        }
    }

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr, TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS, TrainingConfig::WARMUP_LR);
        { typename OPTIMIZER::PARAMETERS op; cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost); op.learning_rate = current_lr; cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice); }
        std::cout << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        T epoch_loss = 0; TI epoch_correct = 0, epoch_correct5 = 0, epoch_total = 0, epoch_failed = 0, epoch_batches = 0;

        std::vector<TI> global_indices;
        if (preload) {
            global_indices.resize(preloaded_train_samples.size());
            std::iota(global_indices.begin(), global_indices.end(), 0);
            std::shuffle(global_indices.begin(), global_indices.end(), data_rng);
        }
        auto shuffled_train_files = train_files;
        if (!preload) std::shuffle(shuffled_train_files.begin(), shuffled_train_files.end(), data_rng);
        TI shard_count = preload ? 1 : static_cast<TI>(shuffled_train_files.size());

        for (TI shard_i = 0; shard_i < shard_count; shard_i++) {
            ParquetShard shard;
            std::vector<TI> shard_indices;
            const std::vector<ParquetSample>* batch_samples;
            const std::vector<TI>* batch_indices;
            TI shard_batches;

            if (preload) {
                batch_samples = &preloaded_train_samples;
                batch_indices = &global_indices;
                shard_batches = preloaded_train_samples.size() / batch_size;
            } else {
                if (!shard.load(shuffled_train_files[shard_i])) continue;
                shard_indices.resize(shard.samples.size());
                std::iota(shard_indices.begin(), shard_indices.end(), 0);
                std::shuffle(shard_indices.begin(), shard_indices.end(), data_rng);
                batch_samples = &shard.samples;
                batch_indices = &shard_indices;
                shard_batches = shard.samples.size() / batch_size;
            }

            for (TI batch_i = 0; batch_i < shard_batches; batch_i++) {
                auto batch_start = std::chrono::high_resolution_clock::now();
                cudaDeviceSynchronize();
                auto t_post_sync = std::chrono::high_resolution_clock::now();
                rlt::zero_gradient(device_cuda, model);
                auto t_post_zg = std::chrono::high_resolution_clock::now();
                decltype(batch_start) t_data_end, t_h2d_end, t_fwd_end, t_loss_end, t_thread_start, t_thread_end, t_memcpy_end, t_pre_step, t_post_step;
                T batch_loss = 0; TI batch_correct = 0, batch_correct5 = 0, batch_valid = 0;

                for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                    TI base_idx = batch_i * batch_size + micro_i * GPU_BATCH;
                    std::atomic<TI> micro_failed{0};
                    std::vector<uint32_t> rng_seeds(GPU_BATCH);
                    for (TI s_i = 0; s_i < GPU_BATCH; s_i++) rng_seeds[s_i] = data_rng();
                    t_thread_start = std::chrono::high_resolution_clock::now();

                    auto process_sample = [&](TI s_i) {
                        auto& sample = (*batch_samples)[(*batch_indices)[base_idx + s_i]];
                        auto decoded = decode_jpeg(sample.image_data, sample.image_size);
                        if (!decoded.valid) {
                            micro_failed++;
                            std::memset(cpu_input_staging.data() + s_i * IMG_ELEMS, 0, IMG_ELEMS * sizeof(T));
                            cpu_labels[s_i] = 0;
                            return;
                        }
                        std::mt19937 local_rng(rng_seeds[s_i]);
                        random_resize_crop(decoded, worker_resize_bufs[s_i].data(), TrainingConfig::IMAGE_SIZE, TrainingConfig::CROP_SCALE_MIN, TrainingConfig::CROP_SCALE_MAX, TrainingConfig::CROP_RATIO_MIN, TrainingConfig::CROP_RATIO_MAX, local_rng);
                        std::uniform_real_distribution<T> flip_dist(0, 1);
                        if (flip_dist(local_rng) < TrainingConfig::HFLIP_PROB) {
                            constexpr TI S = TrainingConfig::IMAGE_SIZE;
                            uint8_t* buf = worker_resize_bufs[s_i].data();
                            for (TI y = 0; y < S; y++) for (TI x = 0; x < S / 2; x++) for (TI c = 0; c < 3; c++)
                                std::swap(buf[(y * S + x) * 3 + c], buf[(y * S + (S - 1 - x)) * 3 + c]);
                        }
                        preprocess_to_buffer(worker_resize_bufs[s_i].data(), cpu_input_staging.data(), s_i);
                        cpu_labels[s_i] = static_cast<TI_CUDA>(sample.label);
                    };

                    {
                        std::vector<std::thread> threads;
                        TI samples_per_worker = (GPU_BATCH + num_workers - 1) / num_workers;
                        for (TI w = 0; w < num_workers; w++) {
                            TI start = w * samples_per_worker;
                            TI end = std::min(start + samples_per_worker, static_cast<TI>(GPU_BATCH));
                            if (start >= GPU_BATCH) break;
                            threads.emplace_back([&, start, end]() {
                                for (TI s_i = start; s_i < end; s_i++) process_sample(s_i);
                            });
                        }
                        for (auto& t : threads) t.join();
                    }
                    epoch_failed += micro_failed;
                    t_thread_end = std::chrono::high_resolution_clock::now();

                    // Copy staging -> pinned -> GPU
                    std::memcpy(cpu_input_pinned, cpu_input_staging.data(), GPU_BATCH * IMG_ELEMS * sizeof(T));
                    t_memcpy_end = std::chrono::high_resolution_clock::now();
                    cudaMemcpy(gpu_input._data, cpu_input_pinned, GPU_BATCH * IMG_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpy(gpu_labels, cpu_labels.data(), GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice);
                    t_h2d_end = std::chrono::high_resolution_clock::now();
                    t_data_end = t_thread_end;

                    // Forward
                    rlt::forward(device_cuda, model, gpu_input, model_buffer, rng_cuda, train_mode);
                    auto output_view = rlt::output(device_cuda, model);
                    t_fwd_end = std::chrono::high_resolution_clock::now();

                    // Loss + gradient on GPU (no D2H round-trip)
                    cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                        output_view._data, gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                        TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING, T(1) / T(batch_size));

                    // Read back only scalar metrics (small transfer)
                    std::vector<T> h_losses(GPU_BATCH);
                    std::vector<TI_CUDA> h_correct(GPU_BATCH), h_correct5(GPU_BATCH);
                    cudaMemcpy(h_losses.data(), gpu_losses, GPU_BATCH * sizeof(T), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_correct.data(), gpu_correct, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_correct5.data(), gpu_correct5, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                    for (TI s_i = 0; s_i < GPU_BATCH; s_i++) {
                        batch_loss += h_losses[s_i];
                        batch_correct += h_correct[s_i];
                        batch_correct5 += h_correct5[s_i];
                    }
                    batch_valid += GPU_BATCH;
                    t_loss_end = std::chrono::high_resolution_clock::now();

                    // Backward
                    rlt::backward_full(device_cuda, model, gpu_input, gpu_d_output, gpu_d_input, model_buffer);
                }
                t_pre_step = std::chrono::high_resolution_clock::now();
                rlt::step(device_cuda, optimizer, model);
                t_post_step = std::chrono::high_resolution_clock::now();
                cudaDeviceSynchronize();

                auto batch_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<T> bd = batch_end - batch_start;
                if (batch_valid > 0) batch_loss /= batch_valid;
                epoch_loss += batch_loss; epoch_correct += batch_correct; epoch_correct5 += batch_correct5; epoch_total += batch_valid;
                epoch_batches++;

                T ba = batch_valid > 0 ? static_cast<T>(batch_correct) / batch_valid * 100.0f : 0;
                T ba5 = batch_valid > 0 ? static_cast<T>(batch_correct5) / batch_valid * 100.0f : 0;
                T sps = batch_valid > 0 ? static_cast<T>(batch_valid) / bd.count() : 0;

                rlt::set_step(device_cpu, device_cpu.logger, global_step);
                rlt::add_scalar(device_cpu, device_cpu.logger, "batch/loss", batch_loss);
                rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top1", ba);
                rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top5", ba5);
                rlt::add_scalar(device_cpu, device_cpu.logger, "batch/img_per_sec", sps);
                rlt::add_scalar(device_cpu, device_cpu.logger, "batch/epoch", static_cast<T>(epoch));

                if (epoch_batches % log_interval == 0) {
                    std::chrono::duration<T> dt_sync = t_post_sync - batch_start, dt_zg = t_post_zg - t_post_sync;
                    std::chrono::duration<T> dt_threads = t_thread_end - t_thread_start;
                    std::chrono::duration<T> dt_memcpy = t_memcpy_end - t_thread_end;
                    std::chrono::duration<T> dt_h2d = t_h2d_end - t_memcpy_end;
                    std::chrono::duration<T> dt_fwd = t_fwd_end - t_h2d_end, dt_loss = t_loss_end - t_fwd_end;
                    std::chrono::duration<T> dt_bwd = t_pre_step - t_loss_end, dt_step = t_post_step - t_pre_step, dt_dsync = batch_end - t_post_step;
                    std::cout << "  [" << epoch_batches << "/" << total_batches << "] " << sps << " img/s zg=" << dt_zg.count()*1000 << "ms thr=" << dt_threads.count()*1000 << "ms cpy=" << dt_memcpy.count()*1000 << "ms h2d=" << dt_h2d.count()*1000 << "ms fwd=" << dt_fwd.count()*1000 << "ms loss=" << dt_loss.count()*1000 << "ms bwd=" << dt_bwd.count()*1000 << "ms step=" << dt_step.count()*1000 << "ms sync=" << dt_dsync.count()*1000 << "ms" << std::endl;
                }
                global_step++;
            }
            if (!preload && (shard_i + 1) % 50 == 0) {
                T running_top1 = epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0;
                std::cout << "  Shards: " << (shard_i + 1) << "/" << shuffled_train_files.size() << " running top1: " << running_top1 << "%" << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<T> ed = epoch_end - epoch_start;
        T train_loss = epoch_batches > 0 ? epoch_loss / epoch_batches : 0;
        T train_top1 = epoch_total > 0 ? static_cast<T>(epoch_correct) / epoch_total * 100.0f : 0;
        T train_top5 = epoch_total > 0 ? static_cast<T>(epoch_correct5) / epoch_total * 100.0f : 0;
        std::cout << "  Train loss=" << train_loss << " top1=" << train_top1 << "% top5=" << train_top5 << "% time=" << ed.count() << "s" << std::endl;

        T epoch_img_per_sec = epoch_total > 0 ? static_cast<T>(epoch_total) / ed.count() : 0;
        std::cout << "  " << epoch_img_per_sec << " img/s avg" << std::endl;

        rlt::set_step(device_cpu, device_cpu.logger, epoch);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/loss", train_loss);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/top1", train_top1);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/top5", train_top5);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/lr", current_lr);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/img_per_sec", epoch_img_per_sec);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/epoch_time_s", ed.count());

        { // Validation
            TI vc = 0, vc5 = 0, vt = 0; T vl = 0;
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
            rlt::Tensor<GPU_D_OUTPUT_SPEC> val_output; rlt::malloc(device_cuda, val_output);
            TI val_shard_count = preload ? 1 : static_cast<TI>(val_files.size());
            for (TI vi = 0; vi < val_shard_count; vi++) {
                ParquetShard vshard;
                const std::vector<ParquetSample>* val_batch;
                TI nvb;
                if (preload) {
                    val_batch = &preloaded_val_samples;
                    nvb = preloaded_val_samples.size() / GPU_BATCH;
                } else {
                    if (!vshard.load(val_files[vi])) continue;
                    val_batch = &vshard.samples;
                    nvb = vshard.samples.size() / GPU_BATCH;
                }
                for (TI vb = 0; vb < nvb; vb++) {
                    {
                        std::vector<std::thread> threads;
                        TI samples_per_worker = (GPU_BATCH + num_workers - 1) / num_workers;
                        for (TI w = 0; w < num_workers; w++) {
                            TI start = w * samples_per_worker;
                            TI end = std::min(start + samples_per_worker, static_cast<TI>(GPU_BATCH));
                            if (start >= GPU_BATCH) break;
                            threads.emplace_back([&, start, end, vb]() {
                                for (TI s_i = start; s_i < end; s_i++) {
                                    auto& s = (*val_batch)[vb * GPU_BATCH + s_i];
                                    auto d = decode_jpeg(s.image_data, s.image_size);
                                    if (!d.valid) { std::memset(cpu_input_staging.data() + s_i * IMG_ELEMS, 0, IMG_ELEMS * sizeof(T)); cpu_labels[s_i] = 0; return; }
                                    center_crop_resize(d, worker_resize_bufs[s_i].data(), TrainingConfig::IMAGE_SIZE, val_scratch_bufs[s_i]);
                                    preprocess_to_buffer(worker_resize_bufs[s_i].data(), cpu_input_staging.data(), s_i);
                                    cpu_labels[s_i] = static_cast<TI_CUDA>(s.label);
                                }
                            });
                        }
                        for (auto& t : threads) t.join();
                    }
                    std::memcpy(cpu_input_pinned, cpu_input_staging.data(), GPU_BATCH * IMG_ELEMS * sizeof(T));
                    cudaMemcpy(gpu_input._data, cpu_input_pinned, GPU_BATCH * IMG_ELEMS * sizeof(T), cudaMemcpyHostToDevice);
                    cudaMemcpy(gpu_labels, cpu_labels.data(), GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice);
                    rlt::evaluate(device_cuda, model, gpu_input, val_output, model_buffer, rng_cuda, eval_mode);
                    cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                        val_output._data, gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                        TrainingConfig::NUM_CLASSES, 0, 0);
                    std::vector<T> h_losses(GPU_BATCH);
                    std::vector<TI_CUDA> h_correct(GPU_BATCH), h_correct5(GPU_BATCH);
                    cudaMemcpy(h_losses.data(), gpu_losses, GPU_BATCH * sizeof(T), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_correct.data(), gpu_correct, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                    cudaMemcpy(h_correct5.data(), gpu_correct5, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                    for (TI s_i = 0; s_i < GPU_BATCH; s_i++) { vl += h_losses[s_i]; vc += h_correct[s_i]; vc5 += h_correct5[s_i]; vt++; }
                }
            }
            rlt::free(device_cuda, val_output);
            T val_loss = vt > 0 ? vl / vt : 0;
            T val_top1 = vt > 0 ? static_cast<T>(vc) / vt * 100.0f : 0;
            T val_top5 = vt > 0 ? static_cast<T>(vc5) / vt * 100.0f : 0;
            std::cout << "  Val loss=" << val_loss << " top1=" << val_top1 << "% top5=" << val_top5 << "%" << std::endl;
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/loss", val_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/top1", val_top1);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/top5", val_top5);
        }

        if ((epoch + 1) % TrainingConfig::CHECKPOINT_INTERVAL == 0 || epoch == TrainingConfig::NUM_EPOCHS - 1) {
            rlt::copy(device_cuda, device_cpu, model, model_cpu);
            std::string ckpt_dir = logdir + "/checkpoints";
            fs::create_directories(ckpt_dir);
            std::string ckpt = ckpt_dir + "/resnet18_cuda_epoch_" + std::to_string(epoch) + ".h5";
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
    cudaFree(gpu_labels); cudaFree(gpu_losses); cudaFree(gpu_correct); cudaFree(gpu_correct5);
    cudaFreeHost(cpu_input_pinned);
    rlt::free(device_cpu, device_cpu.logger);
    std::cout << "Training complete!" << std::endl;
    return 0;
}
