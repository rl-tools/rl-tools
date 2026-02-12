// ResNet-18 Inference: Load a JPEG image, preprocess, and run inference
// Usage: resnet_inference <image.jpg> <model.h5>

#include <rl_tools/operations/cpu.h>
#include <rl_tools/containers/tensor/operations_generic.h>
#include <rl_tools/containers/tensor/operations_cpu.h>
#include <rl_tools/nn/layers/dense/operations_generic.h>
#include <rl_tools/nn/layers/conv2d/operations_generic.h>
#include <rl_tools/nn/layers/max_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/avg_pool2d/operations_generic.h>
#include <rl_tools/nn/layers/resnet_block/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>

// Persist
#include <rl_tools/nn/layers/dense/persist.h>
#include <rl_tools/nn/layers/conv2d/persist.h>
#include <rl_tools/nn/layers/max_pool2d/persist.h>
#include <rl_tools/nn/layers/avg_pool2d/persist.h>
#include <rl_tools/nn/layers/resnet_block/persist.h>
#include <rl_tools/nn_models/sequential/persist.h>

// Shared model definition
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

namespace rlt = rl_tools;
using DEVICE = rlt::devices::DefaultCPU;
using T = double;
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = DEVICE::index_t;

using INPUT_SHAPE = rlt::nn_models::resnet18::INPUT_SHAPE<TYPE_POLICY, TI>;
using CAPABILITY = rlt::nn::capability::Forward<>;
using RESNET18 = rlt::nn_models::resnet18::MODEL<TYPE_POLICY, TI, CAPABILITY>;

static constexpr TI TARGET_SIZE = 224;

int main(int argc, char* argv[]) {
    if(argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <image> <model.h5> <classes.txt>" << std::endl;
        std::cerr << "  image       - Input image (JPEG, PNG, BMP, etc.)" << std::endl;
        std::cerr << "  model.h5    - HDF5 file with ResNet-18 weights (resnet18_test_data.h5)" << std::endl;
        std::cerr << "  classes.txt - ImageNet class names, one per line (imagenet-1k-classes.txt)" << std::endl;
        return 1;
    }

    const std::string image_path = argv[1];
    const std::string model_path = argv[2];
    const std::string classes_path = argv[3];

    // ======================== Load class names ========================
    std::vector<std::string> class_names;
    {
        std::ifstream ifs(classes_path);
        if(!ifs.is_open()) {
            std::cerr << "Error: Failed to open class names file: " << classes_path << std::endl;
            return 1;
        }
        std::string line;
        while(std::getline(ifs, line)) {
            class_names.push_back(line);
        }
        if(class_names.size() != 1000) {
            std::cerr << "Warning: Expected 1000 class names, got " << class_names.size() << std::endl;
        }
    }

    // ======================== Load and preprocess image ========================
    int img_w, img_h, img_channels;
    unsigned char* img_data = stbi_load(image_path.c_str(), &img_w, &img_h, &img_channels, 3); // Force 3 channels (RGB)
    if(!img_data) {
        std::cerr << "Error: Failed to load image: " << image_path << std::endl;
        std::cerr << "  stbi error: " << stbi_failure_reason() << std::endl;
        return 1;
    }
    std::cout << "Loaded image: " << image_path << " (" << img_w << "x" << img_h << "x" << img_channels << ")" << std::endl;

    // Resize to 224x224 using stb_image_resize2 (bilinear)
    std::vector<unsigned char> resized(TARGET_SIZE * TARGET_SIZE * 3);
    stbir_resize_uint8_linear(
        img_data, img_w, img_h, 0,
        resized.data(), TARGET_SIZE, TARGET_SIZE, 0,
        STBIR_RGB
    );
    stbi_image_free(img_data);

    std::cout << "Resized to " << TARGET_SIZE << "x" << TARGET_SIZE << std::endl;

    // ======================== Initialize device and model ========================
    DEVICE device;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 0);

    RESNET18 model;
    typename RESNET18::template Buffer<true> buffer;
    rlt::malloc(device, model);
    rlt::malloc(device, buffer);

    // Load model weights
    std::cout << "Loading model from: " << model_path << std::endl;
    auto file = HighFive::File(model_path, HighFive::File::ReadOnly);
    auto model_group = rlt::get_group(device, file, "model");
    bool load_success = rlt::load(device, model, model_group);
    if(!load_success) {
        std::cerr << "Error: Failed to load model weights" << std::endl;
        return 1;
    }
    std::cout << "Model loaded successfully" << std::endl;

    // ======================== Prepare input tensor ========================
    // Convert image to NHWC float tensor with ImageNet normalization:
    //   pixel = (pixel_uint8 / 255.0 - mean) / std
    rlt::Tensor<rlt::tensor::Specification<T, TI, INPUT_SHAPE>> input;
    rlt::malloc(device, input);

    using FLAT_INPUT_SHAPE = rlt::tensor::Shape<TI, 1, TARGET_SIZE, TARGET_SIZE, 3>;
    auto input_4d = rlt::view_memory<FLAT_INPUT_SHAPE>(device, input);

    for(TI h = 0; h < TARGET_SIZE; h++) {
        for(TI w = 0; w < TARGET_SIZE; w++) {
            for(TI c = 0; c < 3; c++) {
                T pixel = static_cast<T>(resized[(h * TARGET_SIZE + w) * 3 + c]) / 255.0;
                T normalized = (pixel - rlt::nn_models::resnet18::IMAGENET_MEAN[c]) / rlt::nn_models::resnet18::IMAGENET_STD[c];
                rlt::set(device, input_4d, normalized, (TI)0, h, w, c);
            }
        }
    }

    // ======================== Run inference ========================
    using OUTPUT_SHAPE = typename RESNET18::OUTPUT_SHAPE;
    rlt::Tensor<rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>> output;
    rlt::malloc(device, output);

    rlt::Mode<rlt::mode::Evaluation<>> eval_mode;
    rlt::evaluate(device, model, input, output, buffer, rng, eval_mode);

    // ======================== Output results ========================
    auto output_flat = rlt::view_memory<rlt::tensor::Shape<TI, 1000>>(device, output);

    struct Prediction {
        TI class_idx;
        T logit;
    };
    std::vector<Prediction> predictions(1000);
    for(TI i = 0; i < 1000; i++) {
        predictions[i] = {i, rlt::get(device, output_flat, i)};
    }
    std::sort(predictions.begin(), predictions.end(), [](const Prediction& a, const Prediction& b) {
        return a.logit > b.logit;
    });

    // Softmax probabilities
    T max_logit = predictions[0].logit;
    T sum_exp = 0;
    for(TI i = 0; i < 1000; i++) {
        sum_exp += std::exp(predictions[i].logit - max_logit);
    }

    std::cout << "\nTop-5 predictions:" << std::endl;
    for(int k = 0; k < 5; k++) {
        T prob = std::exp(predictions[k].logit - max_logit) / sum_exp;
        TI idx = predictions[k].class_idx;
        std::string name = idx < class_names.size() ? class_names[idx] : "???";
        std::cout << "  " << (k+1) << ". " << name
                  << " (class " << idx << ")"
                  << "  logit: " << predictions[k].logit
                  << "  prob: " << (prob * 100.0) << "%" << std::endl;
    }

    // Cleanup
    rlt::free(device, model);
    rlt::free(device, buffer);
    rlt::free(device, input);
    rlt::free(device, output);

    return 0;
}
