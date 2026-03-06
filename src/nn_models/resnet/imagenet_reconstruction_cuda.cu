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
#include <fstream>

#define RL_TOOLS_DEBUG_CUDA_SYNC

#include "imagenet_pipeline.h"
#include "imagenet_kernels.cuh"

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;
namespace fs = std::filesystem;

// --- Precision setup (same as training) ---

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

// --- Configuration ---

#ifndef CUT_LAYER
#define CUT_LAYER 9
#endif
constexpr TI_CUDA CUT_LAYER_IDX = CUT_LAYER;

#define MICRO_BATCH_SIZE 64
constexpr TI_CUDA GPU_BATCH = MICRO_BATCH_SIZE;

constexpr TI IMAGE_SIZE = 224;
constexpr TI NUM_EPOCHS = 100;
constexpr float BASE_LR = 0.01;
constexpr float MIN_LR = 1e-5;
constexpr TI WARMUP_EPOCHS = 3;
constexpr float WARMUP_LR = 1e-5;
constexpr float MOMENTUM = 0.9;
constexpr float WEIGHT_DECAY = 1e-4;
constexpr float CROP_SCALE_MIN = 0.08;
constexpr float CROP_SCALE_MAX = 1.0;
constexpr float CROP_RATIO_MIN = 0.75;
constexpr float CROP_RATIO_MAX = 1.3333;
constexpr float HFLIP_PROB = 0.5;

// --- SGD optimizer ---

struct SGDParams: rlt::nn::optimizers::sgd::DefaultParameters<TYPE_POLICY>{
    static constexpr float LEARNING_RATE = BASE_LR;
    static constexpr float MOMENTUM = ::MOMENTUM;
    static constexpr float WEIGHT_DECAY = ::WEIGHT_DECAY;
    static constexpr bool NESTEROV = true;
    static constexpr bool ENABLE_WEIGHT_DECAY = true;
};
using OPTIMIZER_SPEC = rlt::nn::optimizers::sgd::Specification<TYPE_POLICY, TI_CUDA, SGDParams>;
using OPTIMIZER = rlt::nn::optimizers::SGD<OPTIMIZER_SPEC>;
using GPU_CAPABILITY = rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>;
using GPU_INPUT_SHAPE = rlt::tensor::Shape<TI_CUDA, GPU_BATCH, IMAGE_SIZE, IMAGE_SIZE, 3>;

// --- Encoder: prefix of ResNet18 up to CUT_LAYER ---

using FULL_MODULE_CHAIN = rlt::nn_models::resnet18::MODULE_CHAIN<TYPE_POLICY, TI_CUDA>;
using ENCODER_CHAIN = rlt::nn_models::sequential::TakeFirst<CUT_LAYER_IDX + 1, FULL_MODULE_CHAIN>;
using ENCODER_CAPABILITY = rlt::nn::capability::Forward<>;
using ENCODER_CUDA = rlt::nn_models::sequential::Build<ENCODER_CAPABILITY, ENCODER_CHAIN, GPU_INPUT_SHAPE>;

using CPU_TYPE_POLICY = rlt::numeric_types::Policy<float>;
using FULL_MODULE_CHAIN_CPU = rlt::nn_models::resnet18::MODULE_CHAIN<CPU_TYPE_POLICY, TI>;
using CPU_INPUT_SHAPE = rlt::tensor::Shape<TI, GPU_BATCH, IMAGE_SIZE, IMAGE_SIZE, 3>;
using FULL_CPU_CAPABILITY = rlt::nn::capability::Forward<>;
using FULL_MODEL_CPU = rlt::nn_models::sequential::Build<FULL_CPU_CAPABILITY, FULL_MODULE_CHAIN_CPU, CPU_INPUT_SHAPE>;
using ENCODER_CHAIN_CPU = rlt::nn_models::sequential::TakeFirst<CUT_LAYER_IDX + 1, FULL_MODULE_CHAIN_CPU>;
using ENCODER_CPU = rlt::nn_models::sequential::Build<FULL_CPU_CAPABILITY, ENCODER_CHAIN_CPU, CPU_INPUT_SHAPE>;

// --- Decoder: conv2d head that maps encoder features to 3-channel reconstruction ---

using ENCODER_OUTPUT_SHAPE = typename ENCODER_CUDA::OUTPUT_SHAPE;
static constexpr TI_CUDA REPR_HEIGHT = rlt::get<rlt::length(ENCODER_OUTPUT_SHAPE{}) - 3>(ENCODER_OUTPUT_SHAPE{});
static constexpr TI_CUDA REPR_WIDTH = rlt::get<rlt::length(ENCODER_OUTPUT_SHAPE{}) - 2>(ENCODER_OUTPUT_SHAPE{});
static constexpr TI_CUDA REPR_CHANNELS = rlt::get_last(ENCODER_OUTPUT_SHAPE{});

template<typename TP, typename TI_T>
using DECODER_CONV1_CONFIG = rlt::nn::layers::conv2d::Configuration<
    TP, TI_T, 64, 3, 3, 1, 1, 1, 1,
    rlt::nn::activation_functions::ActivationFunction::RELU,
    rlt::nn::layers::conv2d::Normalization::BATCH_NORM>;
template<typename TP, typename TI_T>
using DECODER_CONV2_CONFIG = rlt::nn::layers::conv2d::Configuration<
    TP, TI_T, 32, 3, 3, 1, 1, 1, 1,
    rlt::nn::activation_functions::ActivationFunction::RELU,
    rlt::nn::layers::conv2d::Normalization::BATCH_NORM>;
template<typename TP, typename TI_T>
using DECODER_CONV3_CONFIG = rlt::nn::layers::conv2d::Configuration<
    TP, TI_T, 3, 3, 3, 1, 1, 1, 1,
    rlt::nn::activation_functions::ActivationFunction::IDENTITY,
    rlt::nn::layers::conv2d::Normalization::NONE>;

using DECODER_CHAIN = rlt::nn_models::sequential::Module<
    rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV1_CONFIG<TYPE_POLICY, TI_CUDA>>,
    rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV2_CONFIG<TYPE_POLICY, TI_CUDA>>,
    rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV3_CONFIG<TYPE_POLICY, TI_CUDA>>
>;
using DECODER_CUDA = rlt::nn_models::sequential::Build<GPU_CAPABILITY, DECODER_CHAIN, ENCODER_OUTPUT_SHAPE>;

// --- CUDA kernels ---

template<typename T_SRC>
__global__ void bilinear_downsample_kernel(
    const T_SRC* __restrict__ src, T_SRC* __restrict__ dst,
    int src_h, int src_w, int dst_h, int dst_w
) {
    int n = blockIdx.z;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    if (oy >= dst_h || ox >= dst_w) return;

    float sy = ((float)oy + 0.5f) * (float)src_h / (float)dst_h - 0.5f;
    float sx = ((float)ox + 0.5f) * (float)src_w / (float)dst_w - 0.5f;

    int x0 = (int)floorf(sx), y0 = (int)floorf(sy);
    int x1 = x0 + 1, y1 = y0 + 1;
    float fx = sx - (float)x0, fy = sy - (float)y0;
    x0 = max(0, min(x0, src_w-1)); x1 = max(0, min(x1, src_w-1));
    y0 = max(0, min(y0, src_h-1)); y1 = max(0, min(y1, src_h-1));

    const T_SRC* src_base = src + (size_t)n * src_h * src_w * 3;
    T_SRC* dst_pixel = dst + ((size_t)n * dst_h * dst_w + oy * dst_w + ox) * 3;

    for (int c = 0; c < 3; c++) {
        float v = (1.f-fx)*(1.f-fy)*(float)src_base[(y0*src_w + x0)*3 + c]
                + fx*(1.f-fy)*(float)src_base[(y0*src_w + x1)*3 + c]
                + (1.f-fx)*fy*(float)src_base[(y1*src_w + x0)*3 + c]
                + fx*fy*(float)src_base[(y1*src_w + x1)*3 + c];
        dst_pixel[c] = (T_SRC)v;
    }
}

template<typename T_ACT, typename T_GRAD, typename TI_T>
__global__ void mse_loss_gradient_kernel(
    const T_ACT* __restrict__ pred,
    const T_ACT* __restrict__ target,
    T_GRAD* __restrict__ d_pred,
    float* __restrict__ losses,
    TI_T num_elements,
    float loss_weight
) {
    TI_T n = blockIdx.x;
    const T_ACT* p = pred + (size_t)n * num_elements;
    const T_ACT* t = target + (size_t)n * num_elements;
    T_GRAD* d = d_pred + (size_t)n * num_elements;

    float sum_sq = 0;
    for (TI_T i = 0; i < num_elements; i++) {
        float diff = (float)p[i] - (float)t[i];
        sum_sq += diff * diff;
        d[i] = (T_GRAD)(2.0f * diff * loss_weight / (float)num_elements);
    }
    losses[n] = sum_sq / (float)num_elements;
}

template<typename TI_T>
__global__ void reduce_loss_kernel(
    const float* __restrict__ losses, float* __restrict__ acc_loss, TI_T n
) {
    float sl = 0;
    for (TI_T i = 0; i < n; i++) sl += losses[i];
    *acc_loss += sl;
}

// --- Utilities ---

template <auto LAYER_I = 0, typename SOURCE_DEVICE, typename TARGET_DEVICE, typename SOURCE_SPEC, typename TARGET_SPEC>
void copy_prefix(SOURCE_DEVICE& source_device, TARGET_DEVICE& target_device, const rlt::nn_models::sequential::ModuleForward<SOURCE_SPEC>& source, rlt::nn_models::sequential::ModuleForward<TARGET_SPEC>& target) {
    if constexpr(LAYER_I < TARGET_SPEC::NUM_LAYERS) {
        rlt::copy(source_device, target_device, rlt::nn_models::sequential::layer<LAYER_I>(source), rlt::nn_models::sequential::layer<LAYER_I>(target));
        copy_prefix<LAYER_I + 1>(source_device, target_device, source, target);
    }
}

constexpr int NUM_DECODE_WORKERS = 3;
constexpr int NJ_PIPE_PER_WORKER = 4;
constexpr int NUM_DECODE_SLOTS = 6;

int main(int argc, char* argv[]) {
    auto now_tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::ostringstream ts_ss;
    ts_ss << std::put_time(std::localtime(&now_tt), "%Y-%m-%dT%H-%M-%SZ");
    const std::string timestamp_dir = ts_ss.str();
    std::string logdir_prefix = "runs_reconstruction";
    std::string checkpoint_path = "";
    std::string binary_dir = "";
    bool mmap_populate = true;
    TI batch_size = 256;
    TI log_interval = 10;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--binary-dir" && i + 1 < argc) binary_dir = argv[++i];
        else if (arg == "--logdir" && i + 1 < argc) logdir_prefix = argv[++i];
        else if (arg == "--checkpoint" && i + 1 < argc) checkpoint_path = argv[++i];
        else if (arg == "--batch-size" && i + 1 < argc) batch_size = std::stoi(argv[++i]);
        else if (arg == "--log-interval" && i + 1 < argc) log_interval = std::stoi(argv[++i]);
        else if (arg == "--mmap-populate") mmap_populate = true;
        else if (arg == "--no-mmap-populate") mmap_populate = false;
        else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " --binary-dir DIR --checkpoint PATH [--batch-size N] [--log-interval N] [--logdir PREFIX]\n"
                      << "  Trains a reconstruction decoder on frozen ResNet18 representations.\n"
                      << "  CUT_LAYER=" << CUT_LAYER_IDX << " (compile-time, repr " << REPR_HEIGHT << "x" << REPR_WIDTH << "x" << REPR_CHANNELS << ")\n";
            return 0;
        }
    }
    std::string logdir = (fs::path(logdir_prefix) / ("cut" + std::to_string(CUT_LAYER_IDX) + "_" + timestamp_dir)).string();
    if (binary_dir.empty()) { std::cerr << "--binary-dir is required" << std::endl; return 1; }
    if (checkpoint_path.empty()) { std::cerr << "--checkpoint is required" << std::endl; return 1; }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;

    std::cout << "=== ImageNet ResNet-18 Reconstruction ===" << std::endl;
    std::cout << "CUT_LAYER=" << CUT_LAYER_IDX << " repr=" << REPR_HEIGHT << "x" << REPR_WIDTH << "x" << REPR_CHANNELS << std::endl;
    std::cout << "Batch=" << batch_size << " micro=" << GPU_BATCH << " accum=" << num_micro_batches << std::endl;

    // --- Load binary dataset ---
    BinaryDataset train_bin;
    std::vector<Sample> train_samples;
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        std::cout << "Loading binary dataset from " << binary_dir << "..." << std::flush;
        if (!train_bin.load(binary_dir + "/train.bin", mmap_populate)) { std::cerr << "\nFailed to load " << binary_dir << "/train.bin" << std::endl; return 1; }
        train_bin.populate_samples(train_samples);
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << " " << train_samples.size() << " train in "
                  << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    }
    TI total_train_samples = train_samples.size();

    // --- Init devices ---
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);
    rlt::init(device_cpu, device_cpu.logger, fs::path(logdir));
    std::cout << "TensorBoard logdir: " << logdir << std::endl;

    std::mt19937 data_rng(42);
    DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda); rlt::init(device_cuda, rng_cuda, 42);

    // --- Load encoder from checkpoint ---
    ENCODER_CUDA encoder;
    typename ENCODER_CUDA::template Buffer<true> encoder_buffer;
    rlt::malloc(device_cuda, encoder); rlt::malloc(device_cuda, encoder_buffer);
    {
        FULL_MODEL_CPU full_model_cpu;
        rlt::malloc(device_cpu, full_model_cpu);
        auto file = HighFive::File(checkpoint_path, HighFive::File::ReadOnly);
        auto mg = rlt::get_group(device_cpu, file, "model");
        rlt::load(device_cpu, full_model_cpu, mg);

        ENCODER_CPU encoder_cpu;
        rlt::malloc(device_cpu, encoder_cpu);
        copy_prefix(device_cpu, device_cpu, full_model_cpu, encoder_cpu);
        rlt::copy(device_cpu, device_cuda, encoder_cpu, encoder);
        rlt::free(device_cpu, encoder_cpu);
        rlt::free(device_cpu, full_model_cpu);
        std::cout << "Loaded encoder (layers 0.." << CUT_LAYER_IDX << ") from " << checkpoint_path << std::endl;
    }

    // --- Init decoder + optimizer ---
    OPTIMIZER optimizer;
    DECODER_CUDA decoder;
    typename DECODER_CUDA::template Buffer<true> decoder_buffer;
    rlt::malloc(device_cuda, optimizer); rlt::malloc(device_cuda, decoder); rlt::malloc(device_cuda, decoder_buffer);
    rlt::init(device_cuda, optimizer);
    {
        // Init decoder weights on CPU, transfer to GPU
        using DECODER_CHAIN_CPU = rlt::nn_models::sequential::Module<
            rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV1_CONFIG<CPU_TYPE_POLICY, TI>>,
            rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV2_CONFIG<CPU_TYPE_POLICY, TI>>,
            rlt::nn::layers::conv2d::BindConfiguration<DECODER_CONV3_CONFIG<CPU_TYPE_POLICY, TI>>
        >;
        using ENCODER_CPU_OUTPUT_SHAPE = typename ENCODER_CPU::OUTPUT_SHAPE;
        using DECODER_CPU_TYPE = rlt::nn_models::sequential::Build<rlt::nn::capability::Gradient<rlt::nn::parameters::SGD>, DECODER_CHAIN_CPU, ENCODER_CPU_OUTPUT_SHAPE>;
        DECODER_CPU_TYPE decoder_cpu;
        rlt::malloc(device_cpu, decoder_cpu);
        DEVICE_CPU::SPEC::RANDOM::ENGINE<> rng_cpu; rlt::malloc(device_cpu, rng_cpu); rlt::init(device_cpu, rng_cpu, 123);
        rlt::init_weights(device_cpu, decoder_cpu, rng_cpu);
        rlt::copy(device_cpu, device_cuda, decoder_cpu, decoder);
        rlt::free(device_cpu, decoder_cpu);
        rlt::free(device_cpu, rng_cpu);
    }
    rlt::reset_optimizer_state(device_cuda, optimizer, decoder);

    // --- GPU tensors ---
    using GPU_INPUT_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, GPU_INPUT_SHAPE>;
    rlt::Tensor<GPU_INPUT_SPEC> gpu_input; rlt::malloc(device_cuda, gpu_input);

    using REPR_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, ENCODER_OUTPUT_SHAPE>;
    rlt::Tensor<REPR_SPEC> repr; rlt::malloc(device_cuda, repr);

    using TARGET_SHAPE = rlt::tensor::Shape<TI_CUDA, GPU_BATCH, REPR_HEIGHT, REPR_WIDTH, 3>;
    using TARGET_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, TARGET_SHAPE>;
    rlt::Tensor<TARGET_SPEC> gpu_target; rlt::malloc(device_cuda, gpu_target);

    using D_OUTPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, TARGET_SHAPE>;
    rlt::Tensor<D_OUTPUT_SPEC> gpu_d_output; rlt::malloc(device_cuda, gpu_d_output);

    using D_REPR_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, ENCODER_OUTPUT_SHAPE>;
    rlt::Tensor<D_REPR_SPEC> gpu_d_repr; rlt::malloc(device_cuda, gpu_d_repr);

    constexpr TI_CUDA REPR_ELEMENTS = REPR_HEIGHT * REPR_WIDTH * 3;

    float* gpu_losses; CUDA_CHECK(cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(float)));
    float* gpu_acc_loss; CUDA_CHECK(cudaMalloc(&gpu_acc_loss, sizeof(float)));

    // --- Decode pipeline setup ---
    DecodeWorkerCtx<NJ_PIPE_PER_WORKER> decode_workers[NUM_DECODE_WORKERS];
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        NVJPEG_CHECK(nvjpegCreateSimple(&ctx.handle));
        NVJPEG_CHECK(nvjpegDecoderCreate(ctx.handle, NVJPEG_BACKEND_HARDWARE, &ctx.decoder));
        NVJPEG_CHECK(nvjpegDecodeParamsCreate(ctx.handle, &ctx.params));
        NVJPEG_CHECK(nvjpegDecodeParamsSetOutputFormat(ctx.params, NVJPEG_OUTPUT_RGBI));
        for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
            NVJPEG_CHECK(nvjpegDecoderStateCreate(ctx.handle, ctx.decoder, &ctx.states[p]));
            NVJPEG_CHECK(nvjpegBufferPinnedCreate(ctx.handle, nullptr, &ctx.pinned[p]));
            NVJPEG_CHECK(nvjpegBufferDeviceCreate(ctx.handle, nullptr, &ctx.device_buf[p]));
            NVJPEG_CHECK(nvjpegJpegStreamCreate(ctx.handle, &ctx.jpeg_streams[p]));
            NVJPEG_CHECK(nvjpegStateAttachPinnedBuffer(ctx.states[p], ctx.pinned[p]));
            NVJPEG_CHECK(nvjpegStateAttachDeviceBuffer(ctx.states[p], ctx.device_buf[p]));
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
              << " slots=" << NUM_DECODE_SLOTS << " batch=" << GPU_BATCH << std::endl;

    TSQueue<int> submit_queue, ready_queue;

    // --- Pre-parse JPEG dimensions ---
    std::vector<ImgDims> train_dims(train_samples.size());
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto parse_range = [&](const std::vector<Sample>& samples, ImgDims* dims, size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                dims[i] = jpeg_dimensions(samples[i].image_data, samples[i].image_size);
            }
        };
        TI nw = std::min((TI)std::thread::hardware_concurrency(), (TI)32);
        if(nw <= 0) nw = 1;
        std::vector<std::thread> threads;
        size_t n = train_samples.size();
        size_t chunk = (n + nw - 1) / nw;
        for (TI t = 0; t < nw; t++) {
            size_t b = t * chunk, e = std::min(b + chunk, n);
            if (b < e) threads.emplace_back(parse_range, std::cref(train_samples), train_dims.data(), b, e);
        }
        for (auto& t : threads) t.join();
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Pre-parsed JPEG dimensions: " << train_dims.size() << " train in "
                  << std::fixed << std::setprecision(1) << std::chrono::duration<double>(t1 - t0).count() << "s" << std::endl;
    }

    // --- Decode worker threads ---
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
                    const ImageInfo& im = slot.cpu_info[s_i];
                    if(im.img_w <= 0 || im.img_h <= 0 || im.img_w > NVJPEG_MAX_DIM || im.img_h > NVJPEG_MAX_DIM){
                        continue;
                    }
                    int p = s_i % NJ_PIPE_PER_WORKER;
                    NVJPEG_CHECK(nvjpegJpegStreamParse(ctx.handle, slot.jpeg_data[s_i], slot.jpeg_size[s_i], 0, 0, ctx.jpeg_streams[p]));
                    NVJPEG_CHECK(nvjpegDecodeJpegHost(ctx.handle, ctx.decoder, ctx.states[p], ctx.params, ctx.jpeg_streams[p]));
                    NVJPEG_CHECK(nvjpegDecodeJpegTransferToDevice(ctx.handle, ctx.decoder, ctx.states[p], ctx.jpeg_streams[p], ctx.cuda_stream));
                    NVJPEG_CHECK(nvjpegDecodeJpegDevice(ctx.handle, ctx.decoder, ctx.states[p], &slot.nj_dest[s_i], ctx.cuda_stream));
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

    rlt::Mode<rlt::mode::Default<>> train_mode;
    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    TI global_step = 0;

    for (TI epoch = 0; epoch < NUM_EPOCHS; epoch++) {
        auto epoch_start = std::chrono::high_resolution_clock::now();
        float current_lr = cosine_lr(epoch, NUM_EPOCHS, BASE_LR, MIN_LR, WARMUP_EPOCHS, WARMUP_LR);
        {
            typename OPTIMIZER::PARAMETERS op;
            CUDA_CHECK(cudaMemcpy(&op, optimizer.parameters._data, sizeof(op), cudaMemcpyDeviceToHost));
            op.learning_rate = current_lr;
            CUDA_CHECK(cudaMemcpy(optimizer.parameters._data, &op, sizeof(op), cudaMemcpyHostToDevice));
        }
        std::cout << std::defaultfloat << std::setprecision(6) << "=== Epoch " << epoch << " (lr=" << current_lr << ") ===" << std::endl;

        float epoch_loss = 0; TI epoch_batches = 0;

        std::vector<TI> sample_indices(train_samples.size());
        std::iota(sample_indices.begin(), sample_indices.end(), 0);
        std::shuffle(sample_indices.begin(), sample_indices.end(), data_rng);
        std::vector<TI> valid_sample_indices;
        valid_sample_indices.reserve(sample_indices.size());
        for(TI si : sample_indices){
            if(is_valid_dims(train_dims[si].w, train_dims[si].h)){
                valid_sample_indices.push_back(si);
            }
        }
        TI num_batches = valid_sample_indices.size() / batch_size;
        if(num_batches == 0){ std::cerr << "No valid training batches." << std::endl; return 1; }
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
            rlt::zero_gradient(device_cuda, decoder);
            CUDA_CHECK(cudaMemsetAsync(gpu_acc_loss, 0, sizeof(float), device_cuda.stream));

            for (TI micro_i = 0; micro_i < num_micro_batches; micro_i++) {
                int slot_idx = ready_queue.pop();
                auto& slot = slots[slot_idx];

                CUDA_CHECK(cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0));
                constexpr int BLK = 16, TGT = IMAGE_SIZE;
                train_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                    slot.decode_pool, slot.gpu_info, gpu_input._data, TGT,
                    CROP_SCALE_MIN, CROP_SCALE_MAX,
                    std::log(CROP_RATIO_MIN), std::log(CROP_RATIO_MAX),
                    HFLIP_PROB);
                CUDA_KERNEL_CHECK(device_cuda.stream, "train_crop_normalize");

                // Slot data no longer needed
                CUDA_CHECK(cudaEventRecord(slot.consumed_event, device_cuda.stream));

                // Downsample input to target resolution
                constexpr int BLK_DS = 16;
                bilinear_downsample_kernel<<<dim3((REPR_WIDTH+BLK_DS-1)/BLK_DS,(REPR_HEIGHT+BLK_DS-1)/BLK_DS,GPU_BATCH), dim3(BLK_DS,BLK_DS), 0, device_cuda.stream>>>(
                    gpu_input._data, gpu_target._data, TGT, TGT, REPR_HEIGHT, REPR_WIDTH);
                CUDA_KERNEL_CHECK(device_cuda.stream, "bilinear_downsample");

                // Encoder forward (frozen)
                rlt::evaluate(device_cuda, encoder, gpu_input, repr, encoder_buffer, rng_cuda, eval_mode);

                // Decoder forward (trainable)
                rlt::forward(device_cuda, decoder, repr, decoder_buffer, rng_cuda, train_mode);
                auto decoder_output = rlt::output(device_cuda, decoder);

                // MSE loss + gradient
                mse_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    decoder_output._data, gpu_target._data, gpu_d_output._data, gpu_losses,
                    REPR_ELEMENTS, 1.0f / (float)batch_size);
                CUDA_KERNEL_CHECK(device_cuda.stream, "mse_loss_gradient");
                reduce_loss_kernel<<<1, 1, 0, device_cuda.stream>>>(gpu_losses, gpu_acc_loss, GPU_BATCH);
                CUDA_KERNEL_CHECK(device_cuda.stream, "reduce_loss");

                // Backward through decoder only
                rlt::backward_full(device_cuda, decoder, repr, gpu_d_output, gpu_d_repr, decoder_buffer);

                if (submit_count < total_micro) {
                    TI batch_of = submit_count / num_micro_batches;
                    TI micro_of = submit_count % num_micro_batches;
                    TI base_idx = batch_of * batch_size + micro_of * GPU_BATCH;
                    fill_train_slot(slot_idx, base_idx, valid_sample_indices);
                    submit_queue.push(slot_idx);
                    submit_count++;
                }
            }
            rlt::step(device_cuda, optimizer, decoder);
            CUDA_CHECK(cudaStreamSynchronize(device_cuda.stream));

            auto batch_wall_end = std::chrono::high_resolution_clock::now();
            double wall_ms = std::chrono::duration<double, std::milli>(batch_wall_end - batch_wall_start).count();

            float batch_loss;
            CUDA_CHECK(cudaMemcpy(&batch_loss, gpu_acc_loss, sizeof(float), cudaMemcpyDeviceToHost));
            TI batch_valid = batch_size;
            if (batch_valid > 0) batch_loss /= batch_valid;
            epoch_loss += batch_loss;
            epoch_batches++;

            float sps = static_cast<float>(batch_valid) / (wall_ms * 0.001);

            rlt::set_step(device_cpu, device_cpu.logger, global_step);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/mse_loss", batch_loss);
            rlt::add_scalar(device_cpu, device_cpu.logger, "batch/img_per_sec", sps);

            if (epoch_batches % log_interval == 0) {
                std::cout << std::fixed << std::setprecision(6)
                    << "  [" << epoch_batches << "/" << num_batches << "] mse=" << batch_loss
                    << std::setprecision(0) << " " << sps << " img/s"
                    << std::setprecision(1) << " wall=" << wall_ms << "ms"
                    << std::endl;
            }
            global_step++;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ed = epoch_end - epoch_start;
        float train_loss = epoch_batches > 0 ? epoch_loss / epoch_batches : 0;
        std::cout << "  Train mse=" << train_loss << " time=" << ed.count() << "s" << std::endl;

        rlt::set_step(device_cpu, device_cpu.logger, epoch);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/mse_loss", train_loss);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/lr", current_lr);
        rlt::add_scalar(device_cpu, device_cpu.logger, "train/epoch_time_s", ed.count());

        std::cout << std::endl;
    }

    // --- Shutdown ---
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
        for (int p = 0; p < NJ_PIPE_PER_WORKER; p++) {
            NVJPEG_CHECK(nvjpegJpegStreamDestroy(ctx.jpeg_streams[p])); NVJPEG_CHECK(nvjpegBufferPinnedDestroy(ctx.pinned[p]));
            NVJPEG_CHECK(nvjpegBufferDeviceDestroy(ctx.device_buf[p])); NVJPEG_CHECK(nvjpegJpegStateDestroy(ctx.states[p]));
        }
        NVJPEG_CHECK(nvjpegDecodeParamsDestroy(ctx.params)); NVJPEG_CHECK(nvjpegDecoderDestroy(ctx.decoder)); NVJPEG_CHECK(nvjpegDestroy(ctx.handle));
    }
    CUDA_CHECK(cudaFree(gpu_acc_loss)); CUDA_CHECK(cudaFree(gpu_losses));
    rlt::free(device_cuda, encoder); rlt::free(device_cuda, encoder_buffer);
    rlt::free(device_cuda, decoder); rlt::free(device_cuda, decoder_buffer);
    rlt::free(device_cuda, optimizer);
    rlt::free(device_cuda, gpu_input); rlt::free(device_cuda, repr);
    rlt::free(device_cuda, gpu_target); rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_repr);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, device_cpu.logger);

    return 0;
}
