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

#include <cuda_bf16.h>
#include <nvjpeg.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <cerrno>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

#define RL_TOOLS_DEBUG_CUDA_SYNC

#include "imagenet_pipeline.h"
#include "imagenet_kernels.cuh"

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

using T = __nv_bfloat16;
using TYPE_POLICY = rlt::numeric_types::Policy<float,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Parameter, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Activation, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::Gradient, T>,
    rlt::numeric_types::UseCase<rlt::numeric_types::categories::MasterParameter, float>>;
using T_ACTIVATION = TYPE_POLICY::GET<rlt::numeric_types::categories::Activation>;
using T_GRADIENT = TYPE_POLICY::GET<rlt::numeric_types::categories::Gradient>;
using DEVICE_CPU = rlt::devices::DEVICE_FACTORY<>;
using DEVICE_CUDA = rlt::devices::DEVICE_FACTORY_CUDA<>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

#define MICRO_BATCH_SIZE 64
constexpr TI_CUDA GPU_BATCH = MICRO_BATCH_SIZE;

// --- Training hyperparameters ---
struct TrainingConfig {
    static constexpr TI IMAGE_SIZE = 224;
    static constexpr TI NUM_CLASSES = 1000;
    static constexpr TI NUM_EPOCHS = 600;
    static constexpr float BASE_LR = 0.1;
    static constexpr TI BASE_BATCH_SIZE = 256;
    static constexpr float WARMUP_LR = 1e-5;
    static constexpr TI WARMUP_EPOCHS = 5;
    static constexpr float MIN_LR = 0.0;
    static constexpr float MOMENTUM = 0.9;
    static constexpr float WEIGHT_DECAY = 2e-5;
    static constexpr bool NESTEROV = true;
    static constexpr float LABEL_SMOOTHING = 0.1;
    static constexpr float CROP_SCALE_MIN = 0.08;
    static constexpr float CROP_SCALE_MAX = 1.0;
    static constexpr float CROP_RATIO_MIN = 0.75;
    static constexpr float CROP_RATIO_MAX = 1.3333;
    static constexpr float HFLIP_PROB = 0.5;
    static constexpr TI CHECKPOINT_INTERVAL = 1;
};

struct SGDParams: rlt::nn::optimizers::sgd::DefaultParameters<TYPE_POLICY>{
    static constexpr float LEARNING_RATE = TrainingConfig::BASE_LR;
    static constexpr float MOMENTUM = TrainingConfig::MOMENTUM;
    static constexpr float WEIGHT_DECAY = TrainingConfig::WEIGHT_DECAY;
    static constexpr bool NESTEROV = TrainingConfig::NESTEROV;
    static constexpr bool ENABLE_WEIGHT_DECAY = true;
};

using OPTIMIZER_SPEC = rlt::nn::optimizers::sgd::Specification<TYPE_POLICY, TI_CUDA, SGDParams>;
using OPTIMIZER = rlt::nn::optimizers::SGD<OPTIMIZER_SPEC>;
using GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>;
using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, GPU_BATCH, TrainingConfig::IMAGE_SIZE, TrainingConfig::IMAGE_SIZE, 3>;
using RESNET18_CUDA = rlt::nn_models::sequential::Build<GPU_CAPABILITY, rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI_CUDA>, GPU_INPUT_SHAPE>;
using CPU_TYPE_POLICY = rlt::numeric_types::Policy<float>;
using CPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>;
using CPU_INPUT_SHAPE = rlt::tensor::Shape<TI, GPU_BATCH, TrainingConfig::IMAGE_SIZE, TrainingConfig::IMAGE_SIZE, 3>;
using RESNET18_CPU = rlt::nn_models::sequential::Build<CPU_CAPABILITY, rlt::nn_models::resnet18::MODULE_CHAIN<CPU_TYPE_POLICY, TI>, CPU_INPUT_SHAPE>;
using RESNET18_CPU_INFERENCE = typename RESNET18_CPU::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<>>;

// --- GPU cross-entropy loss + gradient kernel ---
__global__ void cross_entropy_loss_gradient_kernel(
    const T_ACTIVATION* __restrict__ logits,    // [N, C]
    const TI_CUDA* __restrict__ labels, // [N]
    T_GRADIENT* __restrict__ d_logits,        // [N, C]
    float* __restrict__ losses,      // [N]
    TI_CUDA* __restrict__ correct,   // [N] (1 if top-1 correct)
    TI_CUDA* __restrict__ correct5,  // [N] (1 if top-5 correct)
    TI_CUDA num_classes,
    float smoothing,
    float loss_weight
) {
    TI_CUDA i = blockIdx.x;
    const T_ACTIVATION* row = logits + i * num_classes;
    T_GRADIENT* d_row = d_logits + i * num_classes;
    TI_CUDA target = labels[i];

    float max_logit = (float)row[0];
    for (TI_CUDA c = 1; c < num_classes; c++) max_logit = fmaxf(max_logit, (float)row[c]);
    float sum_exp = 0;
    for (TI_CUDA c = 0; c < num_classes; c++) sum_exp += expf((float)row[c] - max_logit);
    float log_sum_exp = max_logit + logf(sum_exp);

    float nll = -((float)row[target] - log_sum_exp);
    float kl_uniform = log_sum_exp;
    for (TI_CUDA c = 0; c < num_classes; c++) kl_uniform -= (float)row[c] / (float)num_classes;
    losses[i] = (1.0f - smoothing) * nll + smoothing * kl_uniform;

    float smooth_weight = smoothing / (float)num_classes;
    for (TI_CUDA c = 0; c < num_classes; c++) {
        float softmax_c = expf((float)row[c] - max_logit) / sum_exp;
        float smooth_target = (c == target) ? ((1.0f - smoothing) + smooth_weight) : smooth_weight;
        d_row[c] = (T_GRADIENT)((softmax_c - smooth_target) * loss_weight);
    }

    // Top-1
    TI_CUDA predicted = 0;
    for (TI_CUDA c = 1; c < num_classes; c++) if ((float)row[c] > (float)row[predicted]) predicted = c;
    correct[i] = (predicted == target) ? 1 : 0;

    // Top-5
    TI_CUDA count = 0;
    float target_logit = (float)row[target];
    for (TI_CUDA c = 0; c < num_classes; c++) {
        if ((float)row[c] > target_logit) count++;
        if (count >= 5) break;
    }
    correct5[i] = (count < 5) ? 1 : 0;
}

// GPU-side metric reduction: sum losses and correct counts across the micro-batch
__global__ void reduce_metrics_kernel(
    const float* __restrict__ losses, const TI_CUDA* __restrict__ correct, const TI_CUDA* __restrict__ correct5,
    float* __restrict__ acc_loss, TI_CUDA* __restrict__ acc_correct, TI_CUDA* __restrict__ acc_correct5,
    TI_CUDA n
) {
    float sl = 0; TI_CUDA sc = 0, sc5 = 0;
    for (TI_CUDA i = 0; i < n; i++) { sl += losses[i]; sc += correct[i]; sc5 += correct5[i]; }
    *acc_loss += sl; *acc_correct += sc; *acc_correct5 += sc5;
}

constexpr int NUM_DECODE_WORKERS = 3;
constexpr int NJ_PIPE_PER_WORKER = 4;
constexpr int NUM_DECODE_SLOTS = 6;

int main(int argc, char* argv[]) {
    auto now_tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ts_ss;
    ts_ss << std::put_time(std::localtime(&now_tt), "%Y-%m-%dT%H-%M-%SZ");
    const std::string timestamp_dir = ts_ss.str();
    std::string logdir_prefix = "runs";
    std::string resume_path = "";
    std::string binary_dir = "";
    bool mmap_populate = true;
    bool skip_warmup = false;
    TI batch_size = 256;
    TI log_interval = 1;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--binary-dir" && i + 1 < argc) binary_dir = argv[++i];
        else if (arg == "--logdir" && i + 1 < argc) logdir_prefix = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) log_interval = std::stoi(argv[++i]);
        else if (arg == "--mmap-populate") mmap_populate = true;
        else if (arg == "--no-mmap-populate") mmap_populate = false;
        else if (arg == "--skip-warmup") skip_warmup = true;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --binary-dir DIR [--batch-size N] [--log-interval N] [--logdir PREFIX] [--resume PATH] [--mmap-populate|--no-mmap-populate] [--skip-warmup]\n"
                      << "  --mmap-populate is enabled by default for maximum I/O performance.\n"
                      << "  --skip-warmup skips the LR warmup epochs.\n";
            return 0;
        }
    }
    std::string logdir = (fs::path(logdir_prefix) / timestamp_dir).string();
    if (binary_dir.empty()) { std::cerr << "--binary-dir is required (use prepare_imagenet.py to create it)" << std::endl; return 1; }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;

    BinaryDataset train_bin, val_bin;
    std::vector<Sample> train_samples, val_samples;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "Loading binary dataset from " << binary_dir << "..." << std::flush;
        if (!train_bin.load(binary_dir + "/train.bin", mmap_populate)) { std::cerr << "\nFailed to load " << binary_dir << "/train.bin" << std::endl; return 1; }
        train_bin.populate_samples(train_samples);
        if (val_bin.load(binary_dir + "/val.bin", mmap_populate))
            val_bin.populate_samples(val_samples);
        else
            std::cerr << "\nWarning: Failed to load " << binary_dir << "/val.bin; continuing without validation set." << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << " " << train_samples.size() << " train, " << val_samples.size() << " val in "
                  << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    }
    TI total_train_samples = train_samples.size();

    float scaled_lr = TrainingConfig::BASE_LR * static_cast<float>(batch_size) / static_cast<float>(TrainingConfig::BASE_BATCH_SIZE);

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
    {
        typename OPTIMIZER::PARAMETERS op;
        CUDA_CHECK(cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost));
        op.learning_rate = scaled_lr;
        CUDA_CHECK(cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice));
    }

    RESNET18_CPU model_cpu;
    RESNET18_CPU_INFERENCE model_cpu_inference;
    rlt::malloc(device_cpu, model_cpu);
    rlt::malloc(device_cpu, model_cpu_inference);
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

    using GPU_INPUT_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input; rlt::malloc(device_cuda, gpu_input);
    using GPU_OUTPUT_SHAPE = typename RESNET18_CUDA::OUTPUT_SHAPE;
    using GPU_D_OUTPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, GPU_OUTPUT_SHAPE>;
    rlt::Tensor<GPU_D_OUTPUT_SPEC> gpu_d_output; rlt::malloc(device_cuda, gpu_d_output);
    using GPU_D_INPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_D_INPUT_SPEC> gpu_d_input; rlt::malloc(device_cuda, gpu_d_input);

    // GPU buffers for loss computation (shared, compute-stream only)
    float* gpu_losses; CUDA_CHECK(cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(float)));
    TI_CUDA* gpu_correct; CUDA_CHECK(cudaMalloc(&gpu_correct, GPU_BATCH * sizeof(TI_CUDA)));
    TI_CUDA* gpu_correct5; CUDA_CHECK(cudaMalloc(&gpu_correct5, GPU_BATCH * sizeof(TI_CUDA)));

    float* gpu_acc_loss; CUDA_CHECK(cudaMalloc(&gpu_acc_loss, sizeof(float)));
    TI_CUDA* gpu_acc_correct; CUDA_CHECK(cudaMalloc(&gpu_acc_correct, sizeof(TI_CUDA)));
    TI_CUDA* gpu_acc_correct5; CUDA_CHECK(cudaMalloc(&gpu_acc_correct5, sizeof(TI_CUDA)));

    cudaEvent_t ev_start, ev_dec, ev_fwd, ev_loss, ev_bwd, ev_step;
    CUDA_CHECK(cudaEventCreate(&ev_start)); CUDA_CHECK(cudaEventCreate(&ev_dec)); CUDA_CHECK(cudaEventCreate(&ev_fwd));
    CUDA_CHECK(cudaEventCreate(&ev_loss));  CUDA_CHECK(cudaEventCreate(&ev_bwd)); CUDA_CHECK(cudaEventCreate(&ev_step));
    struct { double dec, fwd, loss, bwd, step, wall; int n; } prof = {};

    // nvJPEG decode — try HARDWARE backend, fall back to single-phase nvjpegDecode
    DecodeWorkerCtx<NJ_PIPE_PER_WORKER> decode_workers[NUM_DECODE_WORKERS];
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        NVJPEG_CHECK(nvjpegCreateSimple(&ctx.handle));
        if (nvjpegDecoderCreate(ctx.handle, NVJPEG_BACKEND_HARDWARE, &ctx.decoder) != NVJPEG_STATUS_SUCCESS) {
            ctx.use_multiphase = false;
            ctx.decoder = nullptr;
            if (w == 0) std::cout << "nvJPEG: HARDWARE backend not available, using single-phase decode" << std::endl;
        }
        NVJPEG_CHECK(nvjpegDecodeParamsCreate(ctx.handle, &ctx.params));
        NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(ctx.params, NVJPEG_OUTPUT_RGBI));
        if (ctx.use_multiphase) {
            for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
                NVJPEG_CHECK(nvjpegDecoderStateCreate(ctx.handle, ctx.decoder, &ctx.states[p]));
                NVJPEG_CHECK(nvjpegBufferPinnedCreate(ctx.handle, nullptr, &ctx.pinned[p]));
                NVJPEG_CHECK(nvjpegBufferDeviceCreate(ctx.handle, nullptr, &ctx.device_buf[p]));
                NVJPEG_CHECK(nvjpegJpegStreamCreate(ctx.handle, &ctx.jpeg_streams[p]));
                NVJPEG_CHECK(nvjpegStateAttachPinnedBuffer(ctx.states[p], ctx.pinned[p]));
                NVJPEG_CHECK(nvjpegStateAttachDeviceBuffer(ctx.states[p], ctx.device_buf[p]));
            }
        } else {
            NVJPEG_CHECK(nvjpegJpegStateCreate(ctx.handle, &ctx.simple_state));
        }
        CUDA_CHECK(cudaStreamCreate(&ctx.cuda_stream));
    }

    DecodeSlot<TI_CUDA, GPU_BATCH> slots[NUM_DECODE_SLOTS];
    for (int s = 0; s < NUM_DECODE_SLOTS; s++) {
        CUDA_CHECK(cudaMalloc(&slots[s].decode_pool, (size_t)GPU_BATCH * DECODE_IMG_STRIDE));
        CUDA_CHECK(cudaMalloc(&slots[s].gpu_info, GPU_BATCH * sizeof(ImageInfo)));
        CUDA_CHECK(cudaMalloc(&slots[s].gpu_labels, GPU_BATCH * sizeof(TI_CUDA)));
        CUDA_CHECK(cudaEventCreateWithFlags(&slots[s].ready_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&slots[s].consumed_event, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(slots[s].consumed_event, device_cuda.stream));
        CUDA_CHECK(cudaMallocHost(&slots[s].cpu_info, GPU_BATCH * sizeof(ImageInfo)));
        CUDA_CHECK(cudaMallocHost(&slots[s].cpu_labels, GPU_BATCH * sizeof(TI_CUDA)));
        for (TI_CUDA i = 0; i < GPU_BATCH; i++) {
            slots[s].nj_dest[i] = {};
            slots[s].nj_dest[i].channel[0] = slots[s].decode_pool + (size_t)i * DECODE_IMG_STRIDE;
            slots[s].nj_dest[i].pitch[0] = DECODE_PITCH;
        }
    }
    std::cout << "nvJPEG HARDWARE: workers=" << NUM_DECODE_WORKERS << " pipe/worker=" << NJ_PIPE_PER_WORKER
              << " slots=" << NUM_DECODE_SLOTS << " batch=" << GPU_BATCH
              << " pool=" << ((size_t)NUM_DECODE_SLOTS * GPU_BATCH * DECODE_IMG_STRIDE) / (1024*1024) << "MB total" << std::endl;

    TSQueue<int> submit_queue, ready_queue;

    // One-off: pre-extract JPEG dimensions (multithreaded, raw SOF parse)
    std::vector<ImgDims> train_dims(train_samples.size()), val_dims(val_samples.size());
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto parse_range = [&](const std::vector<Sample>& samples, ImgDims* dims, size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                dims[i] = jpeg_dimensions(samples[i].image_data, samples[i].image_size);
            }
        };
        TI nw = std::min((TI)std::thread::hardware_concurrency(), (TI)32);
        if(nw <= 0){
            nw = 1;
        }
        std::vector<std::thread> threads;
        auto launch = [&](const std::vector<Sample>& samples, ImgDims* dims) {
            size_t n = samples.size();
            size_t chunk = (n + nw - 1) / nw;
            for (TI t = 0; t < nw; t++) {
                size_t b = t * chunk, e = std::min(b + chunk, n);
                if (b < e) threads.emplace_back(parse_range, std::cref(samples), dims, b, e);
            }
        };
        launch(train_samples, train_dims.data());
        launch(val_samples, val_dims.data());
        for (auto& t : threads) t.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Pre-parsed JPEG dimensions: " << train_dims.size() << " train + " << val_dims.size() << " val ("
                  << nw << " threads) in " << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    }

    // Decode worker threads: each pulls full slots from submit_queue, decodes on its own stream
    std::thread worker_threads[NUM_DECODE_WORKERS];
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        worker_threads[w] = std::thread([w, &decode_workers, &slots, &submit_queue, &ready_queue]() {
            auto& ctx = decode_workers[w];
            while (true) {
                int slot_idx = submit_queue.pop();
                if (slot_idx < 0) { submit_queue.push(-1); break; }
                auto& slot = slots[slot_idx];
                CUDA_CHECK(cudaStreamWaitEvent(ctx.cuda_stream, slot.consumed_event, 0));
                for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
                    ImageInfo& im = slot.cpu_info[s_i];
                    if(im.img_w <= 0 || im.img_h <= 0 || im.img_w > NVJPEG_MAX_DIM || im.img_h > NVJPEG_MAX_DIM){
                        continue;
                    }
                    if (ctx.use_multiphase) {
                        int p = s_i % NJ_PIPE_PER_WORKER;
                        NVJPEG_CHECK(nvjpegJpegStreamParse(ctx.handle, slot.jpeg_data[s_i], slot.jpeg_size[s_i], 0, 0, ctx.jpeg_streams[p]));
                        NVJPEG_CHECK(nvjpegDecodeJpegHost(ctx.handle, ctx.decoder, ctx.states[p], ctx.params, ctx.jpeg_streams[p]));
                        NVJPEG_CHECK(nvjpegDecodeJpegTransferToDevice(ctx.handle, ctx.decoder, ctx.states[p], ctx.jpeg_streams[p], ctx.cuda_stream));
                        NVJPEG_CHECK(nvjpegDecodeJpegDevice(ctx.handle, ctx.decoder, ctx.states[p], &slot.nj_dest[s_i], ctx.cuda_stream));
                    } else {
                        NVJPEG_CHECK(nvjpegDecode(ctx.handle, ctx.simple_state, slot.jpeg_data[s_i], slot.jpeg_size[s_i], NVJPEG_OUTPUT_RGBI, &slot.nj_dest[s_i], ctx.cuda_stream));
                    }
                }
                CUDA_CHECK(cudaMemcpyAsync(slot.gpu_info, slot.cpu_info, GPU_BATCH * sizeof(ImageInfo), cudaMemcpyHostToDevice, ctx.cuda_stream));
                CUDA_CHECK(cudaMemcpyAsync(slot.gpu_labels, slot.cpu_labels, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice, ctx.cuda_stream));
                CUDA_CHECK(cudaEventRecord(slot.ready_event, ctx.cuda_stream));
                ready_queue.push(slot_idx);
            }
        });
    }

    auto is_valid_dims = [](int w, int h){
        return w > 0 && h > 0 && w <= NVJPEG_MAX_DIM && h <= NVJPEG_MAX_DIM;
    };
    TI invalid_train_total = 0;
    TI invalid_val_total = 0;
    TI invalid_train_parse_or_nonpositive = 0;
    TI invalid_train_oversized = 0;
    TI invalid_val_parse_or_nonpositive = 0;
    TI invalid_val_oversized = 0;
    int oversized_train_max_w = 0, oversized_train_max_h = 0;
    int oversized_val_max_w = 0, oversized_val_max_h = 0;
    for(TI ti = 0; ti < (TI)train_samples.size(); ti++){
        const int w = train_dims[ti].w;
        const int h = train_dims[ti].h;
        if(!is_valid_dims(w, h)){
            invalid_train_total++;
            if(w <= 0 || h <= 0){
                invalid_train_parse_or_nonpositive++;
            }
            else{
                invalid_train_oversized++;
                oversized_train_max_w = std::max(oversized_train_max_w, w);
                oversized_train_max_h = std::max(oversized_train_max_h, h);
            }
        }
    }
    std::vector<TI> valid_val_indices;
    valid_val_indices.reserve(val_samples.size());
    for(TI vi = 0; vi < (TI)val_samples.size(); vi++){
        const int w = val_dims[vi].w;
        const int h = val_dims[vi].h;
        if(is_valid_dims(w, h)){
            valid_val_indices.push_back(vi);
        }
        else{
            invalid_val_total++;
            if(w <= 0 || h <= 0){
                invalid_val_parse_or_nonpositive++;
            }
            else{
                invalid_val_oversized++;
                oversized_val_max_w = std::max(oversized_val_max_w, w);
                oversized_val_max_h = std::max(oversized_val_max_h, h);
            }
        }
    }
    if(invalid_train_total > 0 || invalid_val_total > 0){
        std::cout << "Invalid/oversized images detected: train=" << invalid_train_total
                  << " val=" << invalid_val_total << std::endl;
        std::cout << "  train reasons: parse/non-positive=" << invalid_train_parse_or_nonpositive
                  << " oversized=" << invalid_train_oversized;
        if(invalid_train_oversized > 0){
            std::cout << " (max oversized dims " << oversized_train_max_w << "x" << oversized_train_max_h << ")";
        }
        std::cout << std::endl;
        std::cout << "  val reasons: parse/non-positive=" << invalid_val_parse_or_nonpositive
                  << " oversized=" << invalid_val_oversized;
        if(invalid_val_oversized > 0){
            std::cout << " (max oversized dims " << oversized_val_max_w << "x" << oversized_val_max_h << ")";
        }
        std::cout << std::endl;
    }

    auto fill_train_slot = [&](int slot_idx, TI base_idx, const std::vector<TI>& valid_sample_indices) {
        auto& slot = slots[slot_idx];
        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
            TI si = valid_sample_indices[base_idx + s_i];
            auto& sample = train_samples[si];
            slot.cpu_labels[s_i] = static_cast<TI_CUDA>(sample.label);
            const int w = train_dims[si].w;
            const int h = train_dims[si].h;
            slot.jpeg_data[s_i] = sample.image_data;
            slot.jpeg_size[s_i] = sample.image_size;
            slot.cpu_info[s_i] = {w, h, (uint32_t)data_rng()};
        }
    };
    auto fill_val_slot = [&](int slot_idx, TI base_idx, const std::vector<TI>& valid_indices) {
        auto& slot = slots[slot_idx];
        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
            TI vi = valid_indices[base_idx + s_i];
            auto& s = val_samples[vi];
            slot.cpu_labels[s_i] = static_cast<TI_CUDA>(s.label);
            const int w = val_dims[vi].w;
            const int h = val_dims[vi].h;
            slot.jpeg_data[s_i] = s.image_data;
            slot.jpeg_size[s_i] = s.image_size;
            slot.cpu_info[s_i] = {w, h, 0};
        }
    };

    rlt::Mode<rlt::mode::Default<>> train_mode;
    TI total_batches = total_train_samples / batch_size;
    TI global_step = 0;

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        TI warmup_epochs = skip_warmup ? 0 : TrainingConfig::WARMUP_EPOCHS;
        float current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr, TrainingConfig::MIN_LR, warmup_epochs, TrainingConfig::WARMUP_LR);
        {
            typename OPTIMIZER::PARAMETERS op;
            CUDA_CHECK(cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost));
            op.learning_rate = current_lr;
            CUDA_CHECK(cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice));
        }
        std::cout << std::defaultfloat << std::setprecision(6) << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        float epoch_loss = 0; TI epoch_correct = 0, epoch_correct5 = 0, epoch_total = 0, epoch_batches = 0;

        std::vector<TI> sample_indices(train_samples.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), data_rng);
        std::vector<TI> valid_sample_indices;
        valid_sample_indices.reserve(sample_indices.size());
        TI invalid_train_epoch = 0;
        for(TI si : sample_indices){
            if(is_valid_dims(train_dims[si].w, train_dims[si].h)){
                valid_sample_indices.push_back(si);
            }
            else{
                invalid_train_epoch++;
            }
        }
        if(invalid_train_epoch > 0){
            std::cout << "Epoch " << epoch << ": skipped " << invalid_train_epoch << " invalid/oversized training images before GPU decode" << std::endl;
        }
        TI num_batches = valid_sample_indices.size() / batch_size;
        total_batches = num_batches;
        if(num_batches == 0){
            std::cerr << "No valid training batches available after filtering invalid/oversized images." << std::endl;
            return 1;
        }
        TI total_micro = num_batches * num_micro_batches;

        // Pre-fill decode pipeline
        TI submit_count = 0;
        TI prefill = std::min((TI)NUM_DECODE_SLOTS, total_micro);
        for (TI i = 0; i < prefill; i++) {
            TI batch_of = submit_count / num_micro_batches;
            TI micro_of = submit_count % num_micro_batches;
            TI base_idx = batch_of * batch_size + micro_of * GPU_BATCH;
            fill_train_slot(i, base_idx, valid_sample_indices);
            submit_queue.push(i);
            submit_count++;
        }

        for (TI batch_i = 0; batch_i < num_batches; batch_i++) {
            auto batch_wall_start = std::chrono::high_resolution_clock::now();
            rlt::zero_gradient(device_cuda, model);
            CUDA_CHECK(cudaMemsetAsync(gpu_acc_loss, 0, sizeof(float), device_cuda.stream));
            CUDA_CHECK(cudaMemsetAsync(gpu_acc_correct, 0, sizeof(TI_CUDA), device_cuda.stream));
            CUDA_CHECK(cudaMemsetAsync(gpu_acc_correct5, 0, sizeof(TI_CUDA), device_cuda.stream));
            CUDA_CHECK(cudaEventRecord(ev_start, device_cuda.stream));

            for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                int slot_idx = ready_queue.pop();
                auto& slot = slots[slot_idx];

                CUDA_CHECK(cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0));
                constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                train_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    slot.decode_pool, slot.gpu_info, gpu_input._data, TGT,
                    TrainingConfig::CROP_SCALE_MIN, TrainingConfig::CROP_SCALE_MAX,
                    std::log(TrainingConfig::CROP_RATIO_MIN), std::log(TrainingConfig::CROP_RATIO_MAX),
                    TrainingConfig::HFLIP_PROB);
                CUDA_KERNEL_CHECK(device_cuda.stream, "train_crop_normalize");
                CUDA_CHECK(cudaEventRecord(ev_dec, device_cuda.stream));

                rlt::forward(device_cuda, model, gpu_input, model_buffer, rng_cuda, train_mode);
                auto output_view = rlt::output(device_cuda, model);
                CUDA_CHECK(cudaEventRecord(ev_fwd, device_cuda.stream));

                cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    output_view._data, slot.gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                    TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING, 1.0f / (float)batch_size);
                CUDA_KERNEL_CHECK(device_cuda.stream, "cross_entropy_loss_gradient_kernel(train)");
                reduce_metrics_kernel<<<1, 1, 0, device_cuda.stream>>>(
                    gpu_losses, gpu_correct, gpu_correct5, gpu_acc_loss, gpu_acc_correct, gpu_acc_correct5, GPU_BATCH);
                CUDA_KERNEL_CHECK(device_cuda.stream, "reduce_metrics_kernel(train)");
                CUDA_CHECK(cudaEventRecord(ev_loss, device_cuda.stream));

                // Slot data no longer needed after loss kernel reads gpu_labels
                CUDA_CHECK(cudaEventRecord(slot.consumed_event, device_cuda.stream));

                rlt::backward_full(device_cuda, model, gpu_input, gpu_d_output, gpu_d_input, model_buffer);
                CUDA_CHECK(cudaEventRecord(ev_bwd, device_cuda.stream));

                if (submit_count < total_micro) {
                    TI batch_of = submit_count / num_micro_batches;
                    TI micro_of = submit_count % num_micro_batches;
                    TI base_idx = batch_of * batch_size + micro_of * GPU_BATCH;
                    fill_train_slot(slot_idx, base_idx, valid_sample_indices);
                    submit_queue.push(slot_idx);
                    submit_count++;
                }
            }
            rlt::step(device_cuda, optimizer, model);
            CUDA_CHECK(cudaEventRecord(ev_step, device_cuda.stream));
            CUDA_CHECK(cudaStreamSynchronize(device_cuda.stream));

            auto batch_wall_end = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(batch_wall_end - batch_wall_start).count();

            float batch_loss; TI_CUDA batch_correct, batch_correct5;
            CUDA_CHECK(cudaMemcpy(&batch_loss, gpu_acc_loss, sizeof(float), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&batch_correct, gpu_acc_correct, sizeof(TI_CUDA), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&batch_correct5, gpu_acc_correct5, sizeof(TI_CUDA), cudaMemcpyDeviceToHost));
            TI batch_valid = batch_size;
            if (batch_valid > 0) batch_loss /= batch_valid;
            epoch_loss += batch_loss; epoch_correct += batch_correct; epoch_correct5 += batch_correct5; epoch_total += batch_valid;
            epoch_batches++;

            float ba = static_cast<float>(batch_correct) / batch_valid * 100.f;
            float ba5 = static_cast<float>(batch_correct5) / batch_valid * 100.f;
            float sps = static_cast<float>(batch_valid) / (wall_ms * 0.001);

            float ms_dec, ms_fwd, ms_loss, ms_bwd, ms_step;
            CUDA_CHECK(cudaEventElapsedTime(&ms_dec,  ev_start, ev_dec));
            CUDA_CHECK(cudaEventElapsedTime(&ms_fwd,  ev_dec,   ev_fwd));
            CUDA_CHECK(cudaEventElapsedTime(&ms_loss, ev_fwd,   ev_loss));
            CUDA_CHECK(cudaEventElapsedTime(&ms_bwd,  ev_loss,  ev_bwd));
            CUDA_CHECK(cudaEventElapsedTime(&ms_step, ev_bwd,   ev_step));
            prof.dec += ms_dec; prof.fwd += ms_fwd; prof.loss += ms_loss;
            prof.bwd += ms_bwd; prof.step += ms_step; prof.wall += wall_ms; prof.n++;

            rlt::set_step(device_cpu, device_cpu.logger, global_step);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/loss", batch_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top1", ba);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top5", ba5);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/img_per_sec", sps);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/epoch", static_cast<float>(epoch));

            if (epoch_batches % log_interval == 0 && prof.n > 0) {
                double n = prof.n;
                double gpu_total = (prof.dec + prof.fwd + prof.loss + prof.bwd + prof.step) / n;
                std::cout << std::fixed << std::setprecision(4)
                    << "  [" << epoch_batches << "/" << total_batches << "] loss=" << batch_loss << " top1=" << ba << "% top5=" << ba5 << "%"
                    << std::setprecision(0) << " " << sps << " img/s"
                    << std::setprecision(1) << " | wall=" << prof.wall/n << "ms"
                    << std::endl;
                prof = {};
            }
            global_step++;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ed = epoch_end - epoch_start;
        float train_loss = epoch_batches > 0 ? epoch_loss / epoch_batches : 0;
        float train_top1 = epoch_total > 0 ? static_cast<float>(epoch_correct) / epoch_total * 100.0f : 0;
        float train_top5 = epoch_total > 0 ? static_cast<float>(epoch_correct5) / epoch_total * 100.0f : 0;
        std::cout << "  Train loss=" << train_loss << " top1=" << train_top1 << "% top5=" << train_top5 << "% time=" << ed.count() << "s" << std::endl;

        float epoch_img_per_sec = epoch_total > 0 ? static_cast<float>(epoch_total) / ed.count() : 0;
        std::cout << "  " << epoch_img_per_sec << " img/s avg" << std::endl;

        rlt::set_step(device_cpu, device_cpu.logger, epoch);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/loss", train_loss);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/top1", train_top1);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/top5", train_top5);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/lr", current_lr);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/img_per_sec", epoch_img_per_sec);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/epoch_time_s", ed.count());

        { // Validation (pipelined through same producer)
            TI vc = 0, vc5 = 0, vt = 0; float vl = 0;
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
            rlt::Tensor<GPU_D_OUTPUT_SPEC> val_output; rlt::malloc(device_cuda, val_output);
            TI nvb = valid_val_indices.size() / GPU_BATCH;
            if(nvb == 0){
                std::cerr << "No valid validation batches available after filtering invalid/oversized images." << std::endl;
                rlt::free(device_cuda, val_output);
                return 1;
            }
            TI val_submit = 0;
            TI val_prefill = std::min((TI)NUM_DECODE_SLOTS, nvb);
            for (TI i = 0; i < val_prefill; i++) {
                fill_val_slot(i, val_submit * GPU_BATCH, valid_val_indices);
                submit_queue.push(i);
                val_submit++;
            }
            for (TI vb = 0; vb < nvb; vb++) {
                int slot_idx = ready_queue.pop();
                auto& slot = slots[slot_idx];
                CUDA_CHECK(cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0));
                constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                val_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    slot.decode_pool, slot.gpu_info, gpu_input._data, TGT);
                CUDA_KERNEL_CHECK(device_cuda.stream, "val_crop_normalize");
                rlt::evaluate(device_cuda, model, gpu_input, val_output, model_buffer, rng_cuda, eval_mode);
                cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    val_output._data, slot.gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                    TrainingConfig::NUM_CLASSES, 0, 0);
                CUDA_KERNEL_CHECK(device_cuda.stream, "cross_entropy_loss_gradient_kernel(val)");
                CUDA_CHECK(cudaEventRecord(slot.consumed_event, device_cuda.stream));
                std::vector<float> h_losses(GPU_BATCH);
                std::vector<TI_CUDA> h_correct(GPU_BATCH), h_correct5(GPU_BATCH);
                CUDA_CHECK(cudaMemcpy(h_losses.data(), gpu_losses, GPU_BATCH * sizeof(float), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_correct.data(), gpu_correct, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(h_correct5.data(), gpu_correct5, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost));
                for (TI s_i = 0; s_i < GPU_BATCH; s_i++) { vl += h_losses[s_i]; vc += h_correct[s_i]; vc5 += h_correct5[s_i]; vt++; }
                if (val_submit < nvb) {
                    fill_val_slot(slot_idx, val_submit * GPU_BATCH, valid_val_indices);
                    submit_queue.push(slot_idx);
                    val_submit++;
                }
            }
            rlt::free(device_cuda, val_output);
            float val_loss = vt > 0 ? vl / vt : 0;
            float val_top1 = vt > 0 ? static_cast<float>(vc) / vt * 100.0f : 0;
            float val_top5 = vt > 0 ? static_cast<float>(vc5) / vt * 100.0f : 0;
            std::cout << "  Val loss=" << val_loss << " top1=" << val_top1 << "% top5=" << val_top5 << "%" << std::endl;
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/loss", val_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/top1", val_top1);
            rlt::add_scalar(device_cpu, device_cpu.logger, "val/top5", val_top5);
        }

        if ((epoch + 1) % TrainingConfig::CHECKPOINT_INTERVAL == 0 || epoch == TrainingConfig::NUM_EPOCHS - 1) {
            rlt::copy(device_cuda, device_cpu, model, model_cpu);
            rlt::copy(device_cpu, device_cpu, model_cpu, model_cpu_inference);
            std::string ckpt_dir = logdir + "/checkpoints";
            fs::create_directories(ckpt_dir);
            std::string ckpt = ckpt_dir + "/resnet18_cuda_epoch_" + std::to_string(epoch) + ".h5";
            auto file = HighFive::File(ckpt, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Overwrite);
            auto mg = rlt::create_group(device_cpu, file, "model");
            rlt::save(device_cpu, model_cpu_inference, mg);
            std::cout << "  Checkpoint: " << ckpt << std::endl;
        }
        std::cout << std::endl;
    }

    // Shutdown decode workers (one sentinel cascades via re-push)
    submit_queue.push(-1);
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) worker_threads[w].join();

    for (int s = 0; s < NUM_DECODE_SLOTS; s++) {
        CUDA_CHECK(cudaFree(slots[s].decode_pool)); CUDA_CHECK(cudaFree(slots[s].gpu_info)); CUDA_CHECK(cudaFree(slots[s].gpu_labels));
        CUDA_CHECK(cudaEventDestroy(slots[s].ready_event)); CUDA_CHECK(cudaEventDestroy(slots[s].consumed_event));
        CUDA_CHECK(cudaFreeHost(slots[s].cpu_info)); CUDA_CHECK(cudaFreeHost(slots[s].cpu_labels));
    }
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        CUDA_CHECK(cudaStreamDestroy(ctx.cuda_stream));
        if (ctx.use_multiphase) {
            for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
                NVJPEG_CHECK(nvjpegJpegStreamDestroy(ctx.jpeg_streams[p])); NVJPEG_CHECK(nvjpegBufferPinnedDestroy(ctx.pinned[p]));
                NVJPEG_CHECK(nvjpegBufferDeviceDestroy(ctx.device_buf[p])); NVJPEG_CHECK(nvjpegJpegStateDestroy(ctx.states[p]));
            }
            NVJPEG_CHECK(nvjpegDecoderDestroy(ctx.decoder));
        } else {
            NVJPEG_CHECK(nvjpegJpegStateDestroy(ctx.simple_state));
        }
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(ctx.params)); NVJPEG_CHECK(nvjpegDestroy(ctx.handle));
    }
    CUDA_CHECK(cudaFree(gpu_acc_loss)); CUDA_CHECK(cudaFree(gpu_acc_correct)); CUDA_CHECK(cudaFree(gpu_acc_correct5));
    CUDA_CHECK(cudaEventDestroy(ev_start)); CUDA_CHECK(cudaEventDestroy(ev_dec)); CUDA_CHECK(cudaEventDestroy(ev_fwd));
    CUDA_CHECK(cudaEventDestroy(ev_loss));  CUDA_CHECK(cudaEventDestroy(ev_bwd)); CUDA_CHECK(cudaEventDestroy(ev_step));
    rlt::free(device_cuda, model); rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, optimizer); rlt::free(device_cuda, gpu_input);
    rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_input);
    rlt::free(device_cuda, rng_cuda); rlt::free(device_cpu, model_cpu); rlt::free(device_cpu, model_cpu_inference);
    CUDA_CHECK(cudaFree(gpu_losses)); CUDA_CHECK(cudaFree(gpu_correct)); CUDA_CHECK(cudaFree(gpu_correct5));
    rlt::free(device_cpu, device_cpu.logger);
    std::cout << "Training complete!" << std::endl;
    return 0;
}
