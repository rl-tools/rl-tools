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

// nvJPEG helper: get image dimensions from JPEG header
bool nvjpeg_get_dims(nvjpegHandle_t handle, const uint8_t* data, size_t len, int& w, int& h) {
    int nComp; nvjpegChromaSubsampling_t ss;
    int ws[NVJPEG_MAX_COMPONENT], hs[NVJPEG_MAX_COMPONENT];
    if (nvjpegGetImageInfo(handle, data, len, &nComp, &ss, ws, hs) != NVJPEG_STATUS_SUCCESS) return false;
    w = ws[0]; h = hs[0];
    return w > 0 && h > 0 && w <= NVJPEG_MAX_DIM && h <= NVJPEG_MAX_DIM;
}

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

    // GPU buffers for loss computation (no CPU round-trip needed)
    TI_CUDA* gpu_labels; cudaMalloc(&gpu_labels, GPU_BATCH * sizeof(TI_CUDA));
    T* gpu_losses; cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(T));
    TI_CUDA* gpu_correct; cudaMalloc(&gpu_correct, GPU_BATCH * sizeof(TI_CUDA));
    TI_CUDA* gpu_correct5; cudaMalloc(&gpu_correct5, GPU_BATCH * sizeof(TI_CUDA));

    std::vector<TI_CUDA> cpu_labels(GPU_BATCH);

    // nvJPEG setup
    nvjpegHandle_t nj_handle; nvjpegJpegState_t nj_state;
    if (nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr, 0, &nj_handle) != NVJPEG_STATUS_SUCCESS) { std::cerr << "nvJPEG init failed" << std::endl; return 1; }
    nvjpegJpegStateCreate(nj_handle, &nj_state);
    nvjpegDecodeBatchedInitialize(nj_handle, nj_state, GPU_BATCH, std::min(4u, (unsigned)GPU_BATCH), NVJPEG_OUTPUT_RGBI);
    uint8_t* decode_pool; cudaMalloc(&decode_pool, (size_t)GPU_BATCH * DECODE_IMG_STRIDE);
    ImageInfo* gpu_info; cudaMalloc(&gpu_info, GPU_BATCH * sizeof(ImageInfo));
    std::vector<ImageInfo> cpu_info(GPU_BATCH);
    std::vector<nvjpegImage_t> nj_dest(GPU_BATCH);
    std::vector<const unsigned char*> nj_ptrs(GPU_BATCH);
    std::vector<size_t> nj_lens(GPU_BATCH);
    for (TI_CUDA i = 0; i < GPU_BATCH; i++) { nj_dest[i] = {}; nj_dest[i].channel[0] = decode_pool + (size_t)i * DECODE_IMG_STRIDE; nj_dest[i].pitch[0] = DECODE_PITCH; }
    std::cout << "nvJPEG: batch=" << GPU_BATCH << " max_dim=" << NVJPEG_MAX_DIM << " pool=" << ((size_t)GPU_BATCH * DECODE_IMG_STRIDE) / (1024*1024) << "MB" << std::endl;

    rlt::Mode<rlt::mode::Default<>> train_mode;
    TI total_batches = total_train_samples / batch_size;
    TI global_step = 0;

    for (TI epoch = 0; epoch < TrainingConfig::NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        T current_lr = cosine_lr(epoch, TrainingConfig::NUM_EPOCHS, scaled_lr, TrainingConfig::MIN_LR, TrainingConfig::WARMUP_EPOCHS, TrainingConfig::WARMUP_LR);
        { typename OPTIMIZER::PARAMETERS op; cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost); op.learning_rate = current_lr; cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice); }
        std::cout << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        T epoch_loss = 0; TI epoch_correct = 0, epoch_correct5 = 0, epoch_total = 0, epoch_failed = 0, epoch_batches = 0;

        std::vector<TI> sample_indices(train_samples.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), data_rng);
        TI num_batches = train_samples.size() / batch_size;

        {
            for (TI batch_i = 0; batch_i < num_batches; batch_i++) {
                auto batch_start = std::chrono::high_resolution_clock::now();
                cudaDeviceSynchronize();
                auto t_post_sync = std::chrono::high_resolution_clock::now();
                rlt::zero_gradient(device_cuda, model);
                auto t_post_zg = std::chrono::high_resolution_clock::now();
                decltype(batch_start) t_decode_start, t_decode_end, t_fwd_end, t_loss_end, t_pre_step, t_post_step;
                T batch_loss = 0; TI batch_correct = 0, batch_correct5 = 0, batch_valid = 0;

                for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                    TI base_idx = batch_i * batch_size + micro_i * GPU_BATCH;
                    t_decode_start = std::chrono::high_resolution_clock::now();

                    {
                        TI micro_failed = 0;
                        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
                            auto& sample = train_samples[sample_indices[base_idx + s_i]];
                            nj_ptrs[s_i] = sample.image_data; nj_lens[s_i] = sample.image_size;
                            cpu_labels[s_i] = static_cast<TI_CUDA>(sample.label);
                            int w, h;
                            if (nvjpeg_get_dims(nj_handle, sample.image_data, sample.image_size, w, h)) {
                                cpu_info[s_i] = {w, h, (uint32_t)data_rng()};
                            } else {
                                cpu_info[s_i] = {0, 0, 0}; nj_ptrs[s_i] = nullptr; nj_lens[s_i] = 0;
                                micro_failed++;
                            }
                        }
                        nvjpegDecodeBatched(nj_handle, nj_state, nj_ptrs.data(), nj_lens.data(), nj_dest.data(), device_cuda.stream);
                        cudaMemcpyAsync(gpu_info, cpu_info.data(), GPU_BATCH * sizeof(ImageInfo), cudaMemcpyHostToDevice, device_cuda.stream);
                        cudaMemcpyAsync(gpu_labels, cpu_labels.data(), GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice, device_cuda.stream);
                        constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                        train_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                            decode_pool, gpu_info, gpu_input._data, TGT,
                            TrainingConfig::CROP_SCALE_MIN, TrainingConfig::CROP_SCALE_MAX,
                            std::log(TrainingConfig::CROP_RATIO_MIN), std::log(TrainingConfig::CROP_RATIO_MAX),
                            TrainingConfig::HFLIP_PROB);
                        epoch_failed += micro_failed;
                    }
                    t_decode_end = std::chrono::high_resolution_clock::now();

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
                    std::chrono::duration<T> dt_decode = t_decode_end - t_decode_start;
                    std::chrono::duration<T> dt_fwd = t_fwd_end - t_decode_end, dt_loss = t_loss_end - t_fwd_end;
                    std::chrono::duration<T> dt_bwd = t_pre_step - t_loss_end, dt_step = t_post_step - t_pre_step, dt_dsync = batch_end - t_post_step;
                    std::cout << "  [" << epoch_batches << "/" << total_batches << "] " << sps << " img/s zg=" << dt_zg.count()*1000 << "ms dec=" << dt_decode.count()*1000 << "ms fwd=" << dt_fwd.count()*1000 << "ms loss=" << dt_loss.count()*1000 << "ms bwd=" << dt_bwd.count()*1000 << "ms step=" << dt_step.count()*1000 << "ms sync=" << dt_dsync.count()*1000 << "ms" << std::endl;
                }
                global_step++;
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
            TI nvb = val_samples.size() / GPU_BATCH;
            for (TI vb = 0; vb < nvb; vb++) {
                for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
                    auto& s = val_samples[vb * GPU_BATCH + s_i];
                    nj_ptrs[s_i] = s.image_data; nj_lens[s_i] = s.image_size;
                    cpu_labels[s_i] = static_cast<TI_CUDA>(s.label);
                    int w, h;
                    if (nvjpeg_get_dims(nj_handle, s.image_data, s.image_size, w, h))
                        cpu_info[s_i] = {w, h, 0};
                    else { cpu_info[s_i] = {0, 0, 0}; nj_ptrs[s_i] = nullptr; nj_lens[s_i] = 0; }
                }
                nvjpegDecodeBatched(nj_handle, nj_state, nj_ptrs.data(), nj_lens.data(), nj_dest.data(), device_cuda.stream);
                cudaMemcpyAsync(gpu_info, cpu_info.data(), GPU_BATCH * sizeof(ImageInfo), cudaMemcpyHostToDevice, device_cuda.stream);
                cudaMemcpyAsync(gpu_labels, cpu_labels.data(), GPU_BATCH * sizeof(TI_CUDA), cudaMemcpyHostToDevice, device_cuda.stream);
                constexpr int BLK = 16, TGT = TrainingConfig::IMAGE_SIZE;
                val_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    decode_pool, gpu_info, gpu_input._data, TGT);
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

    nvjpegJpegStateDestroy(nj_state); nvjpegDestroy(nj_handle);
    cudaFree(decode_pool); cudaFree(gpu_info);
    rlt::free(device_cuda, model); rlt::free(device_cuda, model_buffer);
    rlt::free(device_cuda, optimizer); rlt::free(device_cuda, gpu_input);
    rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_input);
    rlt::free(device_cuda, rng_cuda); rlt::free(device_cpu, model_cpu);
    cudaFree(gpu_labels); cudaFree(gpu_losses); cudaFree(gpu_correct); cudaFree(gpu_correct5);
    rlt::free(device_cpu, device_cpu.logger);
    std::cout << "Training complete!" << std::endl;
    return 0;
}
