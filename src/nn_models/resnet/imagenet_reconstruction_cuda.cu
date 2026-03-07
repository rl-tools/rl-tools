#define RL_TOOLS_OPERATIONS_CPU_MUX_INCLUDE_CUDA
#include <rl_tools/operations/cpu_mux.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_generic.h>
#include <rl_tools/nn/optimizers/sgd/instance/operations_cuda.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/upsample2d/operations_generic.h>
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

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

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
#define CUT_LAYER 2
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

// --- Decoder: upsample2d + conv2d blocks mapping encoder features back to 224x224x3 ---

using ENCODER_OUTPUT_SHAPE = typename ENCODER_CUDA::OUTPUT_SHAPE;
static constexpr TI_CUDA REPR_HEIGHT = rlt::get<rlt::length(ENCODER_OUTPUT_SHAPE{}) - 3>(ENCODER_OUTPUT_SHAPE{});
static constexpr TI_CUDA REPR_WIDTH = rlt::get<rlt::length(ENCODER_OUTPUT_SHAPE{}) - 2>(ENCODER_OUTPUT_SHAPE{});
static constexpr TI_CUDA REPR_CHANNELS = rlt::get_last(ENCODER_OUTPUT_SHAPE{});

constexpr TI_CUDA compute_upsample_stages(TI_CUDA repr_h, TI_CUDA target_h) {
    TI_CUDA stages = 0;
    TI_CUDA h = repr_h;
    while (h < target_h) { h *= 2; stages++; }
    return stages;
}
static constexpr TI_CUDA NUM_UPSAMPLE_STAGES = compute_upsample_stages(REPR_HEIGHT, IMAGE_SIZE);
static_assert(REPR_HEIGHT * (1 << NUM_UPSAMPLE_STAGES) == IMAGE_SIZE, "Encoder spatial dim must be a power-of-2 divisor of IMAGE_SIZE");

namespace decoder_detail {
    template<typename A, typename B>
    struct ConcatModules;
    template<typename... As, typename... Bs>
    struct ConcatModules<rlt::nn_models::sequential::Module<As...>, rlt::nn_models::sequential::Module<Bs...>> {
        using type = rlt::nn_models::sequential::Module<As..., Bs...>;
    };
}

// Recursive decoder chain builder: generates (upsample2d + conv2d) blocks
// Uses int for STAGES/CHANNELS to avoid nvcc issues with dependent NTTPs in partial specializations
// STAGES_REMAINING=1 is the final block (outputs 3 channels, no norm, identity)
// STAGES_REMAINING>1 are intermediate blocks (halve channels, BN, ReLU)
// STAGES_REMAINING=0 means encoder is already at target resolution (just a 1x1 conv to 3ch)
template<typename TP, typename TI_T, int STAGES_REMAINING, int IN_CHANNELS>
struct DecoderChainBuilder {
    static_assert(STAGES_REMAINING > 1);
    static constexpr int OUT_CHANNELS = IN_CHANNELS / 2 < 32 ? 32 : IN_CHANNELS / 2;
    using BLOCK = rlt::nn_models::sequential::Module<
        rlt::nn::layers::upsample2d::BindConfiguration<rlt::nn::layers::upsample2d::Configuration<TP, TI_T, (TI_T)2, (TI_T)2>>,
        rlt::nn::layers::conv2d::BindConfiguration<rlt::nn::layers::conv2d::Configuration<
            TP, TI_T, (TI_T)OUT_CHANNELS, (TI_T)3, (TI_T)3, (TI_T)1, (TI_T)1, (TI_T)1, (TI_T)1,
            rlt::nn::activation_functions::ActivationFunction::RELU,
            rlt::nn::layers::conv2d::Normalization::BATCH_NORM>>>;
    using REST = typename DecoderChainBuilder<TP, TI_T, STAGES_REMAINING - 1, OUT_CHANNELS>::CHAIN;
    using CHAIN = typename decoder_detail::ConcatModules<BLOCK, REST>::type;
};

template<typename TP, typename TI_T, int IN_CHANNELS>
struct DecoderChainBuilder<TP, TI_T, 1, IN_CHANNELS> {
    using CHAIN = rlt::nn_models::sequential::Module<
        rlt::nn::layers::upsample2d::BindConfiguration<rlt::nn::layers::upsample2d::Configuration<TP, TI_T, (TI_T)2, (TI_T)2>>,
        rlt::nn::layers::conv2d::BindConfiguration<rlt::nn::layers::conv2d::Configuration<
            TP, TI_T, (TI_T)3, (TI_T)3, (TI_T)3, (TI_T)1, (TI_T)1, (TI_T)1, (TI_T)1,
            rlt::nn::activation_functions::ActivationFunction::IDENTITY,
            rlt::nn::layers::conv2d::Normalization::NONE>>>;
};

template<typename TP, typename TI_T, int IN_CHANNELS>
struct DecoderChainBuilder<TP, TI_T, 0, IN_CHANNELS> {
    using CHAIN = rlt::nn_models::sequential::Module<
        rlt::nn::layers::conv2d::BindConfiguration<rlt::nn::layers::conv2d::Configuration<
            TP, TI_T, (TI_T)3, (TI_T)1, (TI_T)1, (TI_T)1, (TI_T)1, (TI_T)0, (TI_T)0,
            rlt::nn::activation_functions::ActivationFunction::IDENTITY,
            rlt::nn::layers::conv2d::Normalization::NONE>>>;
};

using DECODER_CHAIN = typename DecoderChainBuilder<TYPE_POLICY, TI_CUDA, (int)NUM_UPSAMPLE_STAGES, (int)REPR_CHANNELS>::CHAIN;
using DECODER_CUDA = rlt::nn_models::sequential::Build<GPU_CAPABILITY, DECODER_CHAIN, ENCODER_OUTPUT_SHAPE>;

// --- CUDA kernels ---

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

// --- Sample image output ---

constexpr int SAMPLE_COLS = 4; // number of image pairs per row

void save_sample_png(const char* path, const T_ACTIVATION* gpu_input_ptr, const T_ACTIVATION* gpu_recon_ptr, int img_size, int num_images) {
    int n = num_images < SAMPLE_COLS ? num_images : SAMPLE_COLS;
    size_t pixels_per_img = (size_t)img_size * img_size * 3;
    size_t total_gpu = (size_t)n * pixels_per_img;
    std::vector<float> input_host(total_gpu), recon_host(total_gpu);
    std::vector<T_ACTIVATION> tmp(total_gpu);
    CUDA_CHECK(cudaMemcpy(tmp.data(), gpu_input_ptr, total_gpu * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total_gpu; i++) input_host[i] = (float)tmp[i];
    CUDA_CHECK(cudaMemcpy(tmp.data(), gpu_recon_ptr, total_gpu * sizeof(T_ACTIVATION), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < total_gpu; i++) recon_host[i] = (float)tmp[i];
    int out_w = n * img_size * 2;
    int out_h = img_size;
    std::vector<uint8_t> rgb(out_w * out_h * 3);
    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float std[3] = {0.229f, 0.224f, 0.225f};
    for (int img_i = 0; img_i < n; img_i++) {
        const float* inp = input_host.data() + img_i * pixels_per_img;
        const float* rec = recon_host.data() + img_i * pixels_per_img;
        int x_off_orig = img_i * img_size * 2;
        int x_off_recon = x_off_orig + img_size;
        for (int y = 0; y < img_size; y++) {
            for (int x = 0; x < img_size; x++) {
                for (int c = 0; c < 3; c++) {
                    size_t src_idx = ((size_t)y * img_size + x) * 3 + c;
                    float v_in = inp[src_idx] * std[c] + mean[c];
                    float v_re = rec[src_idx] * std[c] + mean[c];
                    v_in = v_in < 0.f ? 0.f : (v_in > 1.f ? 1.f : v_in);
                    v_re = v_re < 0.f ? 0.f : (v_re > 1.f ? 1.f : v_re);
                    rgb[((size_t)y * out_w + x_off_orig + x) * 3 + c] = (uint8_t)(v_in * 255.f + 0.5f);
                    rgb[((size_t)y * out_w + x_off_recon + x) * 3 + c] = (uint8_t)(v_re * 255.f + 0.5f);
                }
            }
        }
    }
    stbi_write_png(path, out_w, out_h, 3, rgb.data(), out_w * 3);
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
                      << "  CUT_LAYER=" << CUT_LAYER_IDX << " (repr " << REPR_HEIGHT << "x" << REPR_WIDTH << "x" << REPR_CHANNELS
                      << ", " << NUM_UPSAMPLE_STAGES << " upsample stages)\n";
            return 0;
        }
    }
    std::string logdir = (fs::path(logdir_prefix) / ("cut" + std::to_string(CUT_LAYER_IDX) + "_" + timestamp_dir)).string();
    if (binary_dir.empty()) { std::cerr << "--binary-dir is required" << std::endl; return 1; }
    if (checkpoint_path.empty()) { std::cerr << "--checkpoint is required" << std::endl; return 1; }
    if (batch_size % GPU_BATCH != 0) { std::cerr << "batch-size must be multiple of " << GPU_BATCH << std::endl; return 1; }
    TI num_micro_batches = batch_size / GPU_BATCH;

    std::cout << "=== ImageNet ResNet-18 Reconstruction ===" << std::endl;
    std::cout << "CUT_LAYER=" << CUT_LAYER_IDX << " repr=" << REPR_HEIGHT << "x" << REPR_WIDTH << "x" << REPR_CHANNELS
              << " upsample_stages=" << NUM_UPSAMPLE_STAGES << std::endl;
    std::cout << "Batch=" << batch_size << " micro=" << GPU_BATCH << " accum=" << num_micro_batches << std::endl;

    // --- Load binary dataset ---
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
        using DECODER_CHAIN_CPU = typename DecoderChainBuilder<CPU_TYPE_POLICY, TI, (int)NUM_UPSAMPLE_STAGES, (int)REPR_CHANNELS>::CHAIN;
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

    using DECODER_OUTPUT_SHAPE = typename DECODER_CUDA::OUTPUT_SHAPE;
    using D_OUTPUT_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, DECODER_OUTPUT_SHAPE>;
    rlt::Tensor<D_OUTPUT_SPEC> gpu_d_output; rlt::malloc(device_cuda, gpu_d_output);

    using D_REPR_SPEC = rlt::tensor::Specification<T_GRADIENT, TI_CUDA, ENCODER_OUTPUT_SHAPE>;
    rlt::Tensor<D_REPR_SPEC> gpu_d_repr; rlt::malloc(device_cuda, gpu_d_repr);

    using VAL_OUTPUT_SPEC = rlt::tensor::Specification<T_ACTIVATION, TI_CUDA, DECODER_OUTPUT_SHAPE>;
    rlt::Tensor<VAL_OUTPUT_SPEC> val_output; rlt::malloc(device_cuda, val_output);

    constexpr TI_CUDA OUTPUT_ELEMENTS = IMAGE_SIZE * IMAGE_SIZE * 3;

    float* gpu_losses; CUDA_CHECK(cudaMalloc(&gpu_losses, GPU_BATCH * sizeof(float)));
    float* gpu_acc_loss; CUDA_CHECK(cudaMalloc(&gpu_acc_loss, sizeof(float)));

    // --- Decode pipeline setup ---
    bool use_hw_decode = true;
    DecodeWorkerCtx<NJ_PIPE_PER_WORKER> decode_workers[NUM_DECODE_WORKERS];
    for (int w = 0; w < NUM_DECODE_WORKERS; w++) {
        auto& ctx = decode_workers[w];
        NVJPEG_CHECK(nvjpegCreateSimple(&ctx.handle));
        if (nvjpegDecoderCreate(ctx.handle, NVJPEG_BACKEND_HARDWARE, &ctx.decoder) != NVJPEG_STATUS_SUCCESS) {
            ctx.use_multiphase = false;
            ctx.decoder = nullptr;
            if (w == 0) { std::cout << "nvJPEG: HARDWARE backend not available, using single-phase decode" << std::endl; use_hw_decode = false; }
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
              << " slots=" << NUM_DECODE_SLOTS << " batch=" << GPU_BATCH << std::endl;

    TSQueue<int> submit_queue, ready_queue;

    // --- Pre-parse JPEG dimensions ---
    std::vector<ImgDims> train_dims(train_samples.size()), val_dims(val_samples.size());
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        auto parse_range = [](const std::vector<Sample>& samples, ImgDims* dims, size_t begin, size_t end) {
            for (size_t i = begin; i < end; i++) {
                dims[i] = jpeg_dimensions(samples[i].image_data, samples[i].image_size);
            }
        };
        TI nw = std::min((TI)std::thread::hardware_concurrency(), (TI)32);
        if(nw <= 0) nw = 1;
        auto launch = [&](const std::vector<Sample>& samples, ImgDims* dims) {
            std::vector<std::thread> threads;
            size_t n = samples.size();
            size_t chunk = (n + nw - 1) / nw;
            for (TI t = 0; t < nw; t++) {
                size_t b = t * chunk, e = std::min(b + chunk, n);
                if (b < e) threads.emplace_back(parse_range, std::cref(samples), dims, b, e);
            }
            for (auto& t : threads) t.join();
        };
        launch(train_samples, train_dims.data());
        launch(val_samples, val_dims.data());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::cout << "Pre-parsed JPEG dimensions: " << train_dims.size() << " train, " << val_dims.size() << " val in "
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

    std::vector<TI> valid_val_indices;
    valid_val_indices.reserve(val_samples.size());
    for (TI vi = 0; vi < (TI)val_samples.size(); vi++) {
        if (is_valid_dims(val_dims[vi].w, val_dims[vi].h))
            valid_val_indices.push_back(vi);
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
    auto fill_val_slot = [&](int slot_idx, TI base_idx) {
        auto& slot = slots[slot_idx];
        for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) {
            TI vi = valid_val_indices[base_idx + s_i];
            auto& sample = val_samples[vi];
            slot.cpu_labels[s_i] = static_cast<TI_CUDA>(sample.label);
            slot.jpeg_data[s_i] = sample.image_data;
            slot.jpeg_size[s_i] = sample.image_size;
            slot.cpu_info[s_i] = {val_dims[vi].w, val_dims[vi].h, 0};
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

                // Encoder forward (frozen)
                rlt::evaluate(device_cuda, encoder, gpu_input, repr, encoder_buffer, rng_cuda, eval_mode);

                // Decoder forward (trainable)
                rlt::forward(device_cuda, decoder, repr, decoder_buffer, rng_cuda, train_mode);
                auto decoder_output = rlt::output(device_cuda, decoder);

                // Save sample images every 100 batches (on first micro-batch)
                if (epoch_batches % 100 == 0 && micro_i == 0) {
                    CUDA_CHECK(cudaStreamSynchronize(device_cuda.stream));
                    save_sample_png((std::string("reconstruction_sample_layer_") + std::to_string(CUT_LAYER) + ".png").c_str(), gpu_input._data, decoder_output._data, IMAGE_SIZE, GPU_BATCH);
                }

                // MSE loss + gradient (reconstruct the input image)
                mse_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                    decoder_output._data, gpu_input._data, gpu_d_output._data, gpu_losses,
                    OUTPUT_ELEMENTS, 1.0f / (float)batch_size);
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

        // --- Validation ---
        if (!valid_val_indices.empty()) {
            TI nvb = valid_val_indices.size() / GPU_BATCH;
            if (nvb > 0) {
                float val_loss_sum = 0;
                TI val_total = 0;
                TI val_submit = 0;
                TI val_prefill = std::min((TI)NUM_DECODE_SLOTS, nvb);
                for (TI i = 0; i < val_prefill; i++) {
                    fill_val_slot(i, val_submit * GPU_BATCH);
                    submit_queue.push(i);
                    val_submit++;
                }
                for (TI vb = 0; vb < nvb; vb++) {
                    int slot_idx = ready_queue.pop();
                    auto& slot = slots[slot_idx];
                    CUDA_CHECK(cudaStreamWaitEvent(device_cuda.stream, slot.ready_event, 0));
                    constexpr int BLK = 16, TGT = IMAGE_SIZE;
                    val_crop_normalize<<<dim3((TGT+BLK-1)/BLK,(TGT+BLK-1)/BLK,GPU_BATCH), dim3(BLK,BLK), 0, device_cuda.stream>>>(
                        slot.decode_pool, slot.gpu_info, gpu_input._data, TGT);
                    CUDA_KERNEL_CHECK(device_cuda.stream, "val_crop_normalize");
                    CUDA_CHECK(cudaEventRecord(slot.consumed_event, device_cuda.stream));
                    rlt::evaluate(device_cuda, encoder, gpu_input, repr, encoder_buffer, rng_cuda, eval_mode);
                    rlt::evaluate(device_cuda, decoder, repr, val_output, decoder_buffer, rng_cuda, eval_mode);
                    mse_loss_gradient_kernel<<<GPU_BATCH, 1, 0, device_cuda.stream>>>(
                        val_output._data, gpu_input._data, gpu_d_output._data, gpu_losses,
                        OUTPUT_ELEMENTS, 0.f);
                    CUDA_KERNEL_CHECK(device_cuda.stream, "val_mse_loss");
                    std::vector<float> h_losses(GPU_BATCH);
                    CUDA_CHECK(cudaMemcpy(h_losses.data(), gpu_losses, GPU_BATCH * sizeof(float), cudaMemcpyDeviceToHost));
                    for (TI_CUDA s_i = 0; s_i < GPU_BATCH; s_i++) { val_loss_sum += h_losses[s_i]; val_total++; }
                    if (val_submit < nvb) {
                        fill_val_slot(slot_idx, val_submit * GPU_BATCH);
                        submit_queue.push(slot_idx);
                        val_submit++;
                    }
                }
                float val_mse = val_total > 0 ? val_loss_sum / val_total : 0;
                std::cout << "  Val mse=" << val_mse << std::endl;
                rlt::add_scalar(device_cpu, device_cpu.logger, "val/mse_loss", val_mse);
            }
        }

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
    CUDA_CHECK(cudaFree(gpu_acc_loss)); CUDA_CHECK(cudaFree(gpu_losses));
    rlt::free(device_cuda, encoder); rlt::free(device_cuda, encoder_buffer);
    rlt::free(device_cuda, decoder); rlt::free(device_cuda, decoder_buffer);
    rlt::free(device_cuda, optimizer);
    rlt::free(device_cuda, gpu_input); rlt::free(device_cuda, repr);
    rlt::free(device_cuda, gpu_d_output); rlt::free(device_cuda, gpu_d_repr); rlt::free(device_cuda, val_output);
    rlt::free(device_cuda, rng_cuda);
    rlt::free(device_cpu, device_cpu.logger);

    return 0;
}
