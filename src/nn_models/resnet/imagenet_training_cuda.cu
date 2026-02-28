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

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
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

// --- nvJPEG decode + crop/resize/normalize ---
constexpr int NVJPEG_MAX_DIM = 2048;
constexpr int DECODE_PITCH = NVJPEG_MAX_DIM * 3;
constexpr size_t DECODE_IMG_STRIDE = (size_t)NVJPEG_MAX_DIM * NVJPEG_MAX_DIM * 3;

struct ImageInfo { int img_w, img_h; uint32_t rng_seed; };

__constant__ float d_imagenet_mean[3] = {0.485f, 0.456f, 0.406f};
__constant__ float d_imagenet_std[3]  = {0.229f, 0.224f, 0.225f};

__device__ uint32_t xorshift32(uint32_t& s) { s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s; }
__device__ float rand_uniform(uint32_t& s) { return (float)(xorshift32(s) & 0x7FFFFFFF) / (float)0x7FFFFFFF; }

__device__ void bilinear_normalize(const uint8_t* src, int pitch, int w, int h, float sx, float sy, float* dst) {
    int x0 = (int)floorf(sx), y0 = (int)floorf(sy), x1 = x0 + 1, y1 = y0 + 1;
    float fx = sx - (float)x0, fy = sy - (float)y0;
    x0 = max(0, min(x0, w-1)); x1 = max(0, min(x1, w-1));
    y0 = max(0, min(y0, h-1)); y1 = max(0, min(y1, h-1));
    for (int c = 0; c < 3; c++) {
        float v = (1.f-fx)*(1.f-fy)*(float)src[y0*pitch + x0*3 + c]
                + fx*(1.f-fy)*(float)src[y0*pitch + x1*3 + c]
                + (1.f-fx)*fy*(float)src[y1*pitch + x0*3 + c]
                + fx*fy*(float)src[y1*pitch + x1*3 + c];
        dst[c] = (v / 255.f - d_imagenet_mean[c]) / d_imagenet_std[c];
    }
}

__global__ void train_crop_normalize(
    const uint8_t* __restrict__ pool, const ImageInfo* __restrict__ info, float* __restrict__ out,
    int T, float s_min, float s_max, float lr_min, float lr_max, float flip_p
) {
    int n = blockIdx.z, oy = blockIdx.y * blockDim.y + threadIdx.y, ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= T || ox >= T) return;
    float* dst = out + ((size_t)n * T * T + oy * T + ox) * 3;
    const ImageInfo& im = info[n];
    if (im.img_w <= 0) { dst[0] = dst[1] = dst[2] = 0.f; return; }

    uint32_t rng = im.rng_seed;
    float area = (float)(im.img_w * im.img_h);
    int cx = 0, cy = 0, cw = im.img_w, ch = im.img_h;
    for (int a = 0; a < 10; a++) {
        float ta = (s_min + rand_uniform(rng) * (s_max - s_min)) * area;
        float ratio = expf(lr_min + rand_uniform(rng) * (lr_max - lr_min));
        int tw = (int)(sqrtf(ta * ratio) + 0.5f), th = (int)(sqrtf(ta / ratio) + 0.5f);
        if (tw > 0 && tw <= im.img_w && th > 0 && th <= im.img_h) {
            cx = (int)(rand_uniform(rng) * (float)(im.img_w - tw));
            cy = (int)(rand_uniform(rng) * (float)(im.img_h - th));
            cw = tw; ch = th; goto crop_done;
        }
        rand_uniform(rng); rand_uniform(rng);
    }
    { float r = (float)im.img_w / (float)im.img_h;
      if (r < expf(lr_min)) { cw = im.img_w; ch = (int)((float)im.img_w / expf(lr_min)); }
      else if (r > expf(lr_max)) { ch = im.img_h; cw = (int)((float)im.img_h * expf(lr_max)); }
      cw = min(cw, im.img_w); ch = min(ch, im.img_h);
      cx = (im.img_w - cw) / 2; cy = (im.img_h - ch) / 2; }
crop_done:;
    int hflip = rand_uniform(rng) < flip_p;
    float sx = hflip ? (float)cx + ((float)(T-1-ox) + 0.5f) * (float)cw / (float)T - 0.5f
                     : (float)cx + ((float)ox + 0.5f) * (float)cw / (float)T - 0.5f;
    float sy = (float)cy + ((float)oy + 0.5f) * (float)ch / (float)T - 0.5f;
    bilinear_normalize(pool + (size_t)n * DECODE_IMG_STRIDE, DECODE_PITCH, im.img_w, im.img_h, sx, sy, dst);
}

__global__ void val_crop_normalize(
    const uint8_t* __restrict__ pool, const ImageInfo* __restrict__ info, float* __restrict__ out, int T
) {
    int n = blockIdx.z, oy = blockIdx.y * blockDim.y + threadIdx.y, ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= T || ox >= T) return;
    float* dst = out + ((size_t)n * T * T + oy * T + ox) * 3;
    const ImageInfo& im = info[n];
    if (im.img_w <= 0) { dst[0] = dst[1] = dst[2] = 0.f; return; }
    float scale = 256.f / (float)min(im.img_w, im.img_h);
    float cw = (float)T / scale, ch = (float)T / scale;
    float cx_f = ((float)im.img_w - cw) * 0.5f, cy_f = ((float)im.img_h - ch) * 0.5f;
    float sx = cx_f + ((float)ox + 0.5f) * cw / (float)T - 0.5f;
    float sy = cy_f + ((float)oy + 0.5f) * ch / (float)T - 0.5f;
    bilinear_normalize(pool + (size_t)n * DECODE_IMG_STRIDE, DECODE_PITCH, im.img_w, im.img_h, sx, sy, dst);
}

// GPU-side metric reduction: sum losses and correct counts across the micro-batch
__global__ void reduce_metrics_kernel(
    const T* __restrict__ losses, const TI_CUDA* __restrict__ correct, const TI_CUDA* __restrict__ correct5,
    T* __restrict__ acc_loss, TI_CUDA* __restrict__ acc_correct, TI_CUDA* __restrict__ acc_correct5,
    TI_CUDA n
) {
    T sl = 0; TI_CUDA sc = 0, sc5 = 0;
    for (TI_CUDA i = 0; i < n; i++) { sl += losses[i]; sc += correct[i]; sc5 += correct5[i]; }
    *acc_loss += sl; *acc_correct += sc; *acc_correct5 += sc5;
}

constexpr int NUM_DECODE_WORKERS = 3;
constexpr int NJ_PIPE_PER_WORKER = 4;
constexpr int NUM_DECODE_SLOTS = 6;

template<typename U>
struct TSQueue {
    std::queue<U> q;
    std::mutex mtx;
    std::condition_variable cv;
    void push(U v) { { std::lock_guard<std::mutex> lk(mtx); q.push(v); } cv.notify_one(); }
    U pop() { std::unique_lock<std::mutex> lk(mtx); cv.wait(lk, [&]{ return !q.empty(); }); U v = q.front(); q.pop(); return v; }
};

struct DecodeSlot {
    uint8_t* decode_pool;
    ImageInfo* gpu_info;
    TI_CUDA* gpu_labels;
    cudaEvent_t ready_event;
    cudaEvent_t consumed_event;
    ImageInfo* cpu_info;
    TI_CUDA* cpu_labels;
    const uint8_t* jpeg_data[GPU_BATCH];
    size_t jpeg_size[GPU_BATCH];
    nvjpegImage_t nj_dest[GPU_BATCH];
};

struct Sample {
    const uint8_t* image_data;
    size_t image_size;
    int64_t label;
};

// --- mmap-based binary dataset (produced by prepare_imagenet.py) ---
struct BinaryDataset {
    int fd = -1;
    uint8_t* mapped = nullptr;
    size_t file_size = 0;
    uint64_t num_samples = 0;
    struct IndexEntry { uint64_t offset; uint32_t size; uint32_t label; };
    const IndexEntry* index = nullptr;

    bool load(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) { std::cerr << "Failed to open: " << path << std::endl; return false; }
        struct stat st;
        if (fstat(fd, &st) < 0) { ::close(fd); fd = -1; return false; }
        file_size = st.st_size;
        mapped = static_cast<uint8_t*>(mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        if (mapped == MAP_FAILED) { ::close(fd); fd = -1; mapped = nullptr; return false; }
        madvise(mapped, file_size, MADV_RANDOM);
        num_samples = *reinterpret_cast<const uint64_t*>(mapped);
        index = reinterpret_cast<const IndexEntry*>(mapped + 8);
        return true;
    }
    void populate_samples(std::vector<Sample>& out) const {
        out.resize(num_samples);
        for (uint64_t i = 0; i < num_samples; i++)
            out[i] = {mapped + index[i].offset, index[i].size, static_cast<int64_t>(index[i].label)};
    }
    ~BinaryDataset() {
        if (mapped && mapped != MAP_FAILED) munmap(mapped, file_size);
        if (fd >= 0) ::close(fd);
    }
    BinaryDataset() = default;
    BinaryDataset(const BinaryDataset&) = delete;
    BinaryDataset& operator=(const BinaryDataset&) = delete;
};

T cosine_lr(TI epoch, TI total_epochs, T base_lr, T min_lr, TI warmup_epochs, T warmup_lr) {
    if (epoch < warmup_epochs)
        return warmup_lr + (base_lr - warmup_lr) * static_cast<T>(epoch) / static_cast<T>(warmup_epochs);
    T progress = static_cast<T>(epoch - warmup_epochs) / static_cast<T>(total_epochs - warmup_epochs);
    return min_lr + (base_lr - min_lr) * 0.5f * (1.0f + std::cos(static_cast<T>(M_PI) * progress));
}

int main(int argc, char* argv[]) {
    auto now_tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ts_ss;
    ts_ss << std::put_time(std::gmtime(&now_tt), "%Y-%m-%dT%H-%M-%SZ");
    std::string logdir = "runs/" + ts_ss.str();
    std::string resume_path = "";
    std::string binary_dir = "";
    TI batch_size = 256;
    TI log_interval = 1;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--binary-dir" && i + 1 < argc) binary_dir = argv[++i];
        else if (arg == "--logdir" && i + 1 < argc) logdir = argv[++i];
        else if (arg == "--resume" && i + 1 < argc) resume_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) log_interval = std::stoi(argv[++i]);
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --binary-dir DIR [--batch-size N] [--log-interval N] [--logdir DIR] [--resume PATH]\n";
            return 0;
        }
    }
    if (binary_dir.empty()) { std::cerr << "--binary-dir is required (use prepare_imagenet.py to create it)" << std::endl; return 1; }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;

    BinaryDataset train_bin, val_bin;
    std::vector<Sample> train_samples, val_samples;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "Loading binary dataset from " << binary_dir << "..." << std::flush;
        if (!train_bin.load(binary_dir + "/train.bin")) { std::cerr << "\nFailed to load " << binary_dir << "/train.bin" << std::endl; return 1; }
        train_bin.populate_samples(train_samples);
        if (val_bin.load(binary_dir + "/val.bin"))
            val_bin.populate_samples(val_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << " " << train_samples.size() << " train, " << val_samples.size() << " val in "
                  << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    }
    TI total_train_samples = train_samples.size();

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

    using GPU_INPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input; rlt::malloc(device_cuda, gpu_input);
    using GPU_OUTPUT_SHAPE = typename RESNET18_CUDA::OUTPUT_SHAPE;
    using GPU_D_OUTPUT_SPEC = rlt::tensor::Specification<T, TI_CUDA, GPU_OUTPUT_SHAPE>;
    rlt::Tensor<GPU_D_OUTPUT_SPEC> gpu_d_output; rlt::malloc(device_cuda, gpu_d_output);
    rlt::Tensor<GPU_INPUT_SPEC> gpu_d_input; rlt::malloc(device_cuda, gpu_d_input);

    // GPU buffers for loss computation (shared, compute-stream only)
    T* gpu_losses; cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(T));
    TI_CUDA* gpu_correct; cudaMalloc(&gpu_correct, GPU_BATCH * sizeof(TI_CUDA));
    TI_CUDA* gpu_correct5; cudaMalloc(&gpu_correct5, GPU_BATCH * sizeof(TI_CUDA));

    T* gpu_acc_loss; cudaMalloc(&gpu_acc_loss, sizeof(T));
    TI_CUDA* gpu_acc_correct; cudaMalloc(&gpu_acc_correct, sizeof(TI_CUDA));
    TI_CUDA* gpu_acc_correct5; cudaMalloc(&gpu_acc_correct5, sizeof(TI_CUDA));

    cudaEvent_t ev_start, ev_dec, ev_fwd, ev_loss, ev_bwd, ev_step;
    cudaEventCreate(&ev_start); cudaEventCreate(&ev_dec); cudaEventCreate(&ev_fwd);
    cudaEventCreate(&ev_loss);  cudaEventCreate(&ev_bwd); cudaEventCreate(&ev_step);
    struct { double dec, fwd, loss, bwd, step, wall; int n; } prof = {};

    // nvJPEG HARDWARE backend — one context per decode worker (nvjpeg handle is not thread-safe)
    struct DecodeWorkerCtx {
        nvjpegHandle_t handle;
        nvjpegJpegDecoder_t decoder;
        nvjpegDecodeParams_t params;
        nvjpegJpegState_t states[NJ_PIPE_PER_WORKER];
        nvjpegBufferPinned_t pinned[NJ_PIPE_PER_WORKER];
        nvjpegBufferDevice_t device_buf[NJ_PIPE_PER_WORKER];
        nvjpegJpegStream_t jpeg_streams[NJ_PIPE_PER_WORKER];
        cudaStream_t cuda_stream;
    };
    DecodeWorkerCtx decode_workers[NUM_DECODE_WORKERS];
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        if (nvjpegCreateSimple(&ctx.handle) != NVJPEG_STATUS_SUCCESS) { std::cerr << "nvJPEG init failed for worker " << w << std::endl; return 1; }
        if (nvjpegDecoderCreate(ctx.handle, NVJPEG_BACKEND_HARDWARE, &ctx.decoder) != NVJPEG_STATUS_SUCCESS) { std::cerr << "nvJPEG HARDWARE decoder not available (worker " << w << ")" << std::endl; return 1; }
        nvjpegDecodeParamsCreate(ctx.handle, &ctx.params);
        nvjpegDecodeParamsSetOutputFormat(ctx.params, NVJPEG_OUTPUT_RGBI);
        for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
            nvjpegDecoderStateCreate(ctx.handle, ctx.decoder, &ctx.states[p]);
            nvjpegBufferPinnedCreate(ctx.handle, nullptr, &ctx.pinned[p]);
            nvjpegBufferDeviceCreate(ctx.handle, nullptr, &ctx.device_buf[p]);
            nvjpegJpegStreamCreate(ctx.handle, &ctx.jpeg_streams[p]);
            nvjpegStateAttachPinnedBuffer(ctx.states[p], ctx.pinned[p]);
            nvjpegStateAttachDeviceBuffer(ctx.states[p], ctx.device_buf[p]);
        }
        cudaStreamCreate(&ctx.cuda_stream);
    }

    DecodeSlot slots[NUM_DECODE_SLOTS];
    for (int s = 0; s < NUM_DECODE_SLOTS; s++) {
        cudaMalloc(&slots[s].decode_pool, (size_t)GPU_BATCH * DECODE_IMG_STRIDE);
        cudaMalloc(&slots[s].gpu_info, GPU_BATCH * sizeof(ImageInfo));
        cudaMalloc(&slots[s].gpu_labels, GPU_BATCH * sizeof(TI_CUDA));
        cudaEventCreateWithFlags(&slots[s].ready_event, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&slots[s].consumed_event, cudaEventDisableTiming);
        cudaEventRecord(slots[s].consumed_event, device_cuda.stream);
        cudaMallocHost(&slots[s].cpu_info, GPU_BATCH * sizeof(ImageInfo));
        cudaMallocHost(&slots[s].cpu_labels, GPU_BATCH * sizeof(TI_CUDA));
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
    struct ImgDims { int w, h; };
    std::vector<ImgDims> train_dims(train_samples.size()), val_dims(val_samples.size());
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto jpeg_dimensions = [](const uint8_t* data, size_t len) -> ImgDims {
            size_t pos = 2;
            while (pos + 4 < len) {
                if (data[pos] != 0xFF) break;
                uint8_t marker = data[pos + 1];
                if (marker == 0xD9) break;
                if (marker == 0x00 || (marker >= 0xD0 && marker <= 0xD7)) { pos += 2; continue; }
                uint16_t seg_len = ((uint16_t)data[pos + 2] << 8) | data[pos + 3];
                if ((marker & 0xF0) == 0xC0 && marker != 0xC4 && marker != 0xC8 && marker != 0xCC) {
                    if (pos + 9 <= len) {
                        int h = ((int)data[pos + 5] << 8) | data[pos + 6];
                        int w = ((int)data[pos + 7] << 8) | data[pos + 8];
                        return {w, h};
                    }
                }
                pos += 2 + seg_len;
            }
            return {0, 0};
        };
        auto parse_range = [&](const std::vector<Sample>& samples, ImgDims* dims, size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                dims[i] = jpeg_dimensions(samples[i].image_data, samples[i].image_size);
            }
        };
        TI nw = std::min((TI)std::thread::hardware_concurrency(), (TI)32);
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
                DecodeSlot& slot = slots[slot_idx];
                cudaStreamWaitEvent(ctx.cuda_stream, slot.consumed_event, 0);
                for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
                    int p = s_i % NJ_PIPE_PER_WORKER;
                    nvjpegJpegStreamParse(ctx.handle, slot.jpeg_data[s_i], slot.jpeg_size[s_i], 0, 0, ctx.jpeg_streams[p]);
                    nvjpegDecodeJpegHost(ctx.handle, ctx.decoder, ctx.states[p], ctx.params, ctx.jpeg_streams[p]);
                    nvjpegDecodeJpegTransferToDevice(ctx.handle, ctx.decoder, ctx.states[p], ctx.jpeg_streams[p], ctx.cuda_stream);
                    nvjpegDecodeJpegDevice(ctx.handle, ctx.decoder, ctx.states[p], &slot.nj_dest[s_i], ctx.cuda_stream);
                }
                cudaMemcpyAsync(slot.gpu_info, slot.cpu_info, GPU_BATCH * sizeof(ImageInfo), cudaMemcpyHostToDevice, ctx.cuda_stream);
                cudaMemcpyAsync(slot.gpu_labels, slot.cpu_labels, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice, ctx.cuda_stream);
                cudaEventRecord(slot.ready_event, ctx.cuda_stream);
                ready_queue.push(slot_idx);
            }
        });
    }

    auto fill_train_slot = [&](int slot_idx, TI base_idx, const std::vector<TI>& sample_indices) {
        DecodeSlot& slot = slots[slot_idx];
        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
            TI si = sample_indices[base_idx + s_i];
            auto& sample = train_samples[si];
            slot.jpeg_data[s_i] = sample.image_data;
            slot.jpeg_size[s_i] = sample.image_size;
            slot.cpu_labels[s_i] = static_cast<TI_CUDA>(sample.label);
            slot.cpu_info[s_i] = {train_dims[si].w, train_dims[si].h, (uint32_t)data_rng()};
        }
    };
    auto fill_val_slot = [&](int slot_idx, TI base_idx) {
        DecodeSlot& slot = slots[slot_idx];
        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
            TI vi = base_idx + s_i;
            auto& s = val_samples[vi];
            slot.jpeg_data[s_i] = s.image_data;
            slot.jpeg_size[s_i] = s.image_size;
            slot.cpu_labels[s_i] = static_cast<TI_CUDA>(s.label);
            slot.cpu_info[s_i] = {val_dims[vi].w, val_dims[vi].h, 0};
        }
    };

    rlt::Mode<rlt::mode::Default<>> train_mode;
    TI total_batches = total_train_samples / batch_size;
    TI global_step = 0;

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr, TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS, TrainingConfig::WARMUP_LR);
        { typename OPTIMIZER::PARAMETERS op; cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost); op.learning_rate = current_lr; cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice); }
        std::cout << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        T epoch_loss = 0; TI epoch_correct = 0, epoch_correct5 = 0, epoch_total = 0, epoch_batches = 0;

        std::vector<TI> sample_indices(train_samples.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), data_rng);
        TI num_batches = train_samples.size() / batch_size;
        TI total_micro = num_batches * num_micro_batches;

        // Pre-fill decode pipeline
        TI submit_count = 0;
        TI prefill = std::min((TI)NUM_DECODE_SLOTS, total_micro);
        for (TI i = 0; i < prefill; i++) {
            TI batch_of = submit_count / num_micro_batches;
            TI micro_of = submit_count % num_micro_batches;
            TI base_idx = batch_of * batch_size + micro_of * GPU_BATCH;
            fill_train_slot(i, base_idx, sample_indices);
            submit_queue.push(i);
            submit_count++;
        }

        for (TI batch_i = 0; batch_i < num_batches; batch_i++) {
            auto batch_wall_start = std::chrono::high_resolution_clock::now();
            rlt::zero_gradient(device_cuda, model);
            cudaMemsetAsync(gpu_acc_loss, 0, sizeof(T), device_cuda.stream);
            cudaMemsetAsync(gpu_acc_correct, 0, sizeof(TI_CUDA), device_cuda.stream);
            cudaMemsetAsync(gpu_acc_correct5, 0, sizeof(TI_CUDA), device_cuda.stream);
            cudaEventRecord(ev_start, device_cuda.stream);

            for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                int slot_idx = ready_queue.pop();
                DecodeSlot& slot = slots[slot_idx];

                cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0);
                constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                train_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    slot.decode_pool, slot.gpu_info, gpu_input._data, TGT,
                    TrainingConfig::CROP_SCALE_MIN, TrainingConfig::CROP_SCALE_MAX,
                    std::log(TrainingConfig::CROP_RATIO_MIN), std::log(TrainingConfig::CROP_RATIO_MAX),
                    TrainingConfig::HFLIP_PROB);
                cudaEventRecord(ev_dec, device_cuda.stream);

                rlt::forward(device_cuda, model, gpu_input, model_buffer, rng_cuda, train_mode);
                auto output_view = rlt::output(device_cuda, model);
                cudaEventRecord(ev_fwd, device_cuda.stream);

                cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    output_view._data, slot.gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                    TrainingConfig::NUM_CLASSES, TrainingConfig::LABEL_SMOOTHING, T(1) / T(batch_size));
                reduce_metrics_kernel<<<1, 1, 0, device_cuda.stream>>>(
                    gpu_losses, gpu_correct, gpu_correct5, gpu_acc_loss, gpu_acc_correct, gpu_acc_correct5, GPU_BATCH);
                cudaEventRecord(ev_loss, device_cuda.stream);

                // Slot data no longer needed after loss kernel reads gpu_labels
                cudaEventRecord(slot.consumed_event, device_cuda.stream);

                rlt::backward_full(device_cuda, model, gpu_input, gpu_d_output, gpu_d_input, model_buffer);
                cudaEventRecord(ev_bwd, device_cuda.stream);

                if (submit_count < total_micro) {
                    TI batch_of = submit_count / num_micro_batches;
                    TI micro_of = submit_count % num_micro_batches;
                    TI base_idx = batch_of * batch_size + micro_of * GPU_BATCH;
                    fill_train_slot(slot_idx, base_idx, sample_indices);
                    submit_queue.push(slot_idx);
                    submit_count++;
                }
            }
            rlt::step(device_cuda, optimizer, model);
            cudaEventRecord(ev_step, device_cuda.stream);
            cudaStreamSynchronize(device_cuda.stream);

            auto batch_wall_end = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(batch_wall_end - batch_wall_start).count();

            T batch_loss; TI_CUDA batch_correct, batch_correct5;
            cudaMemcpy(&batch_loss, gpu_acc_loss, sizeof(T), cudaMemcpyDeviceToHost);
            cudaMemcpy(&batch_correct, gpu_acc_correct, sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
            cudaMemcpy(&batch_correct5, gpu_acc_correct5, sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
            TI batch_valid = batch_size;
            if (batch_valid > 0) batch_loss /= batch_valid;
            epoch_loss += batch_loss; epoch_correct += batch_correct; epoch_correct5 += batch_correct5; epoch_total += batch_valid;
            epoch_batches++;

            T ba = static_cast<T>(batch_correct) / batch_valid * 100.f;
            T ba5 = static_cast<T>(batch_correct5) / batch_valid * 100.f;
            T sps = static_cast<T>(batch_valid) / (wall_ms * 0.001);

            float ms_dec, ms_fwd, ms_loss, ms_bwd, ms_step;
            cudaEventElapsedTime(&ms_dec,  ev_start, ev_dec);
            cudaEventElapsedTime(&ms_fwd,  ev_dec,   ev_fwd);
            cudaEventElapsedTime(&ms_loss, ev_fwd,   ev_loss);
            cudaEventElapsedTime(&ms_bwd,  ev_loss,  ev_bwd);
            cudaEventElapsedTime(&ms_step, ev_bwd,   ev_step);
            prof.dec += ms_dec; prof.fwd += ms_fwd; prof.loss += ms_loss;
            prof.bwd += ms_bwd; prof.step += ms_step; prof.wall += wall_ms; prof.n++;

            rlt::set_step(device_cpu, device_cpu.logger, global_step);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/loss", batch_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top1", ba);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/top5", ba5);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/img_per_sec", sps);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/epoch", static_cast<T>(epoch));

            if (epoch_batches % log_interval == 0 && prof.n > 0) {
                double n = prof.n;
                double gpu_total = (prof.dec + prof.fwd + prof.loss + prof.bwd + prof.step) / n;
                std::cout << std::fixed << std::setprecision(1)
                    << "  [" << epoch_batches << "/" << total_batches << "] " << std::setprecision(0) << sps << " img/s"
                    << std::setprecision(1) << " | dec=" << prof.dec/n << " fwd=" << prof.fwd/n << " loss=" << prof.loss/n
                    << " bwd=" << prof.bwd/n << " step=" << prof.step/n
                    << " | gpu=" << gpu_total << " wall=" << prof.wall/n << "ms"
                    << " (avg " << prof.n << ")" << std::endl;
                prof = {};
            }
            global_step++;
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

        { // Validation (pipelined through same producer)
            TI vc = 0, vc5 = 0, vt = 0; T vl = 0;
            rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
            rlt::Tensor<GPU_D_OUTPUT_SPEC> val_output; rlt::malloc(device_cuda, val_output);
            TI nvb = val_samples.size() / GPU_BATCH;
            TI val_submit = 0;
            TI val_prefill = std::min((TI)NUM_DECODE_SLOTS, nvb);
            for (TI i = 0; i < val_prefill; i++) {
                fill_val_slot(i, val_submit * GPU_BATCH);
                submit_queue.push(i);
                val_submit++;
            }
            for (TI vb = 0; vb < nvb; vb++) {
                int slot_idx = ready_queue.pop();
                DecodeSlot& slot = slots[slot_idx];
                cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0);
                constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                val_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    slot.decode_pool, slot.gpu_info, gpu_input._data, TGT);
                rlt::evaluate(device_cuda, model, gpu_input, val_output, model_buffer, rng_cuda, eval_mode);
                cross_entropy_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    val_output._data, slot.gpu_labels, gpu_d_output._data, gpu_losses, gpu_correct, gpu_correct5,
                    TrainingConfig::NUM_CLASSES, 0, 0);
                cudaEventRecord(slot.consumed_event, device_cuda.stream);
                std::vector<T> h_losses(GPU_BATCH);
                std::vector<TI_CUDA> h_correct(GPU_BATCH), h_correct5(GPU_BATCH);
                cudaMemcpy(h_losses.data(), gpu_losses, GPU_BATCH * sizeof(T), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_correct.data(), gpu_correct, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                cudaMemcpy(h_correct5.data(), gpu_correct5, GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyDeviceToHost);
                for (TI s_i = 0; s_i < GPU_BATCH; s_i++) { vl += h_losses[s_i]; vc += h_correct[s_i]; vc5 += h_correct5[s_i]; vt++; }
                if (val_submit < nvb) {
                    fill_val_slot(slot_idx, val_submit * GPU_BATCH);
                    submit_queue.push(slot_idx);
                    val_submit++;
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

    // Shutdown decode workers (one sentinel cascades via re-push)
    submit_queue.push(-1);
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) worker_threads[w].join();

    for (int s = 0; s < NUM_DECODE_SLOTS; s++) {
        cudaFree(slots[s].decode_pool); cudaFree(slots[s].gpu_info); cudaFree(slots[s].gpu_labels);
        cudaEventDestroy(slots[s].ready_event); cudaEventDestroy(slots[s].consumed_event);
        cudaFreeHost(slots[s].cpu_info); cudaFreeHost(slots[s].cpu_labels);
    }
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        cudaStreamDestroy(ctx.cuda_stream);
        for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
            nvjpegJpegStreamDestroy(ctx.jpeg_streams[p]); nvjpegBufferPinnedDestroy(ctx.pinned[p]);
            nvjpegBufferDeviceDestroy(ctx.device_buf[p]); nvjpegJpegStateDestroy(ctx.states[p]);
        }
        nvjpegDecodeParamsDestroy(ctx.params); nvjpegDecoderDestroy(ctx.decoder); nvjpegDestroy(ctx.handle);
    }
    cudaFree(gpu_acc_loss); cudaFree(gpu_acc_correct); cudaFree(gpu_acc_correct5);
    cudaEventDestroy(ev_start); cudaEventDestroy(ev_dec); cudaEventDestroy(ev_fwd);
    cudaEventDestroy(ev_loss);  cudaEventDestroy(ev_bwd); cudaEventDestroy(ev_step);
    rlt::free(device_cuda, model); rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, optimizer); rlt::free(device_cuda, gpu_input);
    rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_input);
    rlt::free(device_cuda, rng_cuda); rlt::free(device_cpu, model_cpu);
    cudaFree(gpu_losses); cudaFree(gpu_correct); cudaFree(gpu_correct5);
    rlt::free(device_cpu, device_cpu.logger);
    std::cout << "Training complete!" << std::endl;
    return 0;
}
