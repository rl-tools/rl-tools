// ResNet-18 CUDA Inference: Load model on CPU, copy to GPU, run cuDNN inference
// Usage: resnet_inference_cuda <image> <model.h5> <classes.txt>

// CPU operations (for loading model/image)
#include <rl_tools/operations/cpu/group_1.h>
#include <rl_tools/operations/cuda/group_1.h>

#include <rl_tools/operations/cpu/group_2.h>
#include <rl_tools/operations/cuda/group_2.h>

#include <rl_tools/operations/cpu/group_3.h>
#include <rl_tools/operations/cuda/group_3.h>

// NN operations
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

// CUDA NN operations (cuDNN)
#include <rl_tools/nn/operations_cuda.h>
#include <rl_tools/nn_models/operations_generic.h>

// Persist (CPU only)
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

// Model definition
#include <rl_tools/nn_models/resnet/resnet.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>

namespace rlt = RL_TOOLS_NAMESPACE_WRAPPER ::rl_tools;

using DEVICE_CPU = rlt::devices::DefaultCPU;
using DEVICE_CUDA = rlt::devices::DefaultCUDA;
using T = float;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE_CPU::index_t;
using TI_CUDA = DEVICE_CUDA::index_t;

using INPUT_SHAPE = rlt::nn_models::resnet18::INPUT_SHAPE<TYPE_POLICY, TI>;
using CAPABILITY = rlt::nn::capability::Forward<>;

using RESNET18_CPU = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI, CAPABILITY>;
using RESNET18_CUDA = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI_CUDA, CAPABILITY>;

static constexpr TI TARGET_SIZE = 224;

int main(int argc, char* argv[]){
    if(argc < 4){
        std::cerr << "Usage: " << argv[0] << " <image> <model.h5> <classes.txt>" << std::endl;
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string model_path = argv[2];
    const std::string classes_path = argv[3];

    // ======================== Load class names ========================
    std::vector<std::string> class_names;
    {
        std::ifstream ifs(classes_path);
        if(!ifs.is_open()){
            std::cerr << "Error: Failed to open class names file: " << classes_path << std::endl;
            return 1;
        }
        std::string line;
        while(std::getline(ifs, line)){
            class_names.push_back(line);
        }
    }

    // ======================== Load and preprocess image ========================
    int img_w, img_h, img_channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_channels, 3);
    if(!img_data){
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << image_path << " (" << img_w << "x" << img_h << "x" << img_channels << ")" << std::endl;

    std::vector<unsigned char> resized(TARGET_SIZE * TARGET_SIZE * 3);
    stbir_resize_uint8_linear(img_data, img_w, img_h, 0, resized.data(), TARGET_SIZE, TARGET_SIZE, 0, STBIR_RGB);
    stbi_image_free(img_data);

    // ======================== Initialize devices ========================
    DEVICE_CPU device_cpu;
    DEVICE_CUDA device_cuda;
    rlt::init(device_cuda);

    // ======================== Load model on CPU ========================
    RESNET18_CPU model_cpu;
    rlt::malloc(device_cpu, model_cpu);

    std::cout << "Loading model from: " << model_path << std::endl;
    auto file = HighFive::File(model_path, HighFive::File::ReadOnly);
    auto model_group = rlt::get_group(device_cpu, file, "model");
    if(!rlt::load(device_cpu, model_cpu, model_group)){
        std::cerr << "Error: Failed to load model weights" << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // ======================== Copy model to GPU ========================
    RESNET18_CUDA model_cuda;
    typename RESNET18_CUDA::template Buffer<true> buffer_cuda;
    rlt::malloc(device_cuda, model_cuda);
    rlt::malloc(device_cuda, buffer_cuda);
    rlt::copy(device_cpu, device_cuda, model_cpu, model_cuda);
    std::cout << "Model copied to GPU" << std::endl;

    // ======================== Prepare input on CPU then copy to GPU ========================
    using INPUT_SHAPE_CPU = rlt::tensor::Shape<TI, 1, TARGET_SIZE, TARGET_SIZE, 3>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE_CPU>> input_cpu;
    rlt::malloc(device_cpu, input_cpu);

    auto input_4d = rlt::view_memory<INPUT_SHAPE_CPU>(device_cpu, input_cpu);
    for(TI h = 0; h < TARGET_SIZE; h++){
        for(TI w = 0; w < TARGET_SIZE; w++){
            for(TI c = 0; c < 3; c++){
                T pixel = static_cast<T>(resized[(h * TARGET_SIZE + w) * 3 + c]) / 255.0f;
                T normalized = (pixel - (T)rlt::nn_models::resnet18::IMAGENET_MEAN[c]) / (T)rlt::nn_models::resnet18::IMAGENET_STD[c];
                rlt::set(device_cpu, input_4d, normalized, (TI)0, h, w, c);
            }
        }
    }

    using INPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, TARGET_SIZE, TARGET_SIZE, 3>;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, INPUT_SHAPE_CUDA>> input_cuda;
    rlt::malloc(device_cuda, input_cuda);
    rlt::copy(device_cpu, device_cuda, input_cpu, input_cuda);

    // ======================== Run GPU inference ========================
    using OUTPUT_SHAPE_CUDA = rlt::tensor::Shape<TI_CUDA, 1, 1000>;
    rlt::Tensor<rlt::tensor::Specification<T, TI_CUDA, OUTPUT_SHAPE_CUDA>> output_cuda;
    rlt::malloc(device_cuda, output_cuda);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    typename DEVICE_CUDA::SPEC::RANDOM::ENGINE<> rng_cuda;
    rlt::malloc(device_cuda, rng_cuda);
    rlt::init(device_cuda, rng_cuda, 0);

    // Warmup
    rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda, eval_mode);
    cudaDeviceSynchronize();

    // Timed inference
    auto start = std::chrono::high_resolution_clock::now();
    constexpr int N_ITERS = 100;
    for(int i = 0; i < N_ITERS; i++){
        rlt::evaluate(device_cuda, model_cuda, input_cuda, output_cuda, buffer_cuda, rng_cuda, eval_mode);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "GPU inference: " << ms / N_ITERS << " ms/image (" << N_ITERS << " iterations)" << std::endl;

    // ======================== Copy output back to CPU ========================
    using OUTPUT_SHAPE_CPU = rlt::tensor::Shape<TI, 1, 1000>;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE_CPU>> output_cpu;
    rlt::malloc(device_cpu, output_cpu);
    rlt::copy(device_cuda, device_cpu, output_cuda, output_cpu);

    // ======================== Output results ========================
    auto output_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device_cpu, output_cpu);

    struct Prediction { TI class_idx; T logit; };
    std::vector<Prediction> predictions(1000);
    for(TI i = 0; i < 1000; i++){
        predictions[i] = {i, rlt::get(device_cpu, output_flat, i)};
    }
    std::sort(predictions.begin(), predictions.end(), [](const Prediction& a, const Prediction& b){ return a.logit > b.logit; });

    T max_logit = predictions[0].logit;
    T sum_exp = 0;
    for(TI i = 0; i < 1000; i++){
        sum_exp += std::exp(predictions[i].logit - max_logit);
    }

    std::cout << "\nTop-5 predictions:" << std::endl;
    for(int k = 0; k < 5; k++){
        T prob = std::exp(predictions[k].logit - max_logit) / sum_exp;
        TI idx = predictions[k].class_idx;
        std::string name = idx < class_names.size() ? class_names[idx] : "???";
        std::cout << "  " << (k+1) << ". " << name
                  << " (class " << idx << ")"
                  << "  logit: " << predictions[k].logit
                  << "  prob: " << (prob * 100.0f) << "%" << std::endl;
    }

    // Cleanup
    rlt::free(device_cpu, model_cpu);
    rlt::free(device_cpu, input_cpu);
    rlt::free(device_cpu, output_cpu);
    rlt::free(device_cuda, model_cuda);
    rlt::free(device_cuda, buffer_cuda);
    rlt::free(device_cuda, input_cuda);
    rlt::free(device_cuda, output_cuda);
    rlt::free(device_cuda, rng_cuda);

    return 0;
}
