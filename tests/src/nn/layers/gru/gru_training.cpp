#include <rl_tools/operations/cpu.h>
#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/layers/embedding/operations_generic.h>
#include <rl_tools/nn/layers/gru/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>
#include "dataset.h"

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DefaultCPU;
using TI = typename DEVICE::index_t;
using T = float;

constexpr TI NUM_CLASSES = 2<<8;
constexpr TI EMBEDDING_DIM = 5;
constexpr TI BATCH_SIZE = 8;
constexpr TI SEQUENCE_LENGTH = 128;
constexpr TI OUTPUT_DIM = 2<<8;

template <TI BATCH_SIZE>
using INPUT_SHAPE_TEMPLATE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE>;

int main() {
    DEVICE device;
    auto rng = rlt::random::default_engine(device.random, 0);

    std::string data_path = "/Users/jonas/Downloads/00c2bfc7-57db-496e-9d5c-d62f8d8119e3.json.gzip";
    std::string dataset_string = load_dataset<TI>(data_path);
    std::vector<std::tuple<std::string, unsigned char>> dataset;
    for(TI offset=0; offset < dataset_string.size() - SEQUENCE_LENGTH - 1; offset += BATCH_SIZE){
        dataset.emplace_back(std::tuple(dataset_string.substr(offset, SEQUENCE_LENGTH), dataset_string[offset + SEQUENCE_LENGTH]));
    }
    std::shuffle(dataset.begin(), dataset.end(), rng);
    std::cout << "Dataset size: " << dataset.size() << std::endl;
    std::cout << "Dataset sample: " << std::endl;
    for(TI i=0; i < 10; i++){
        std::cout << std::get<0>(dataset[i]) << " -> " << std::get<1>(dataset[i]) << std::endl;
    }


    using INPUT_SHAPE = INPUT_SHAPE_TEMPLATE<BATCH_SIZE>;
    using INPUT_SPEC = rlt::tensor::Specification<unsigned char, TI, INPUT_SHAPE>;

    using EMBEDDING_LAYER_SPEC = rlt::nn::layers::embedding::Specification<T, TI, NUM_CLASSES, EMBEDDING_DIM, INPUT_SHAPE_TEMPLATE>;
    using CAPABILITY = rlt::nn::layer_capability::Gradient<rlt::nn::parameters::Adam, BATCH_SIZE>;
    using EMBEDDING_LAYER = rlt::nn::layers::embedding::Layer<CAPABILITY, EMBEDDING_LAYER_SPEC>;

    EMBEDDING_LAYER embedding_layer;
    EMBEDDING_LAYER::Buffer<BATCH_SIZE> embedding_buffer;

    using GRU_SPEC = rlt::nn::layers::gru::Specification<T, TI, SEQUENCE_LENGTH, EMBEDDING_DIM, OUTPUT_DIM, rlt::nn::parameters::Gradient>;
    rlt::nn::layers::gru::Layer<CAPABILITY, GRU_SPEC> gru;
    decltype(gru)::Buffer<BATCH_SIZE> gru_buffer;

    using EMBEDDING_OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, EMBEDDING_LAYER::OUTPUT_DIM>;
    using EMBEDDING_OUTPUT_SPEC = rlt::tensor::Specification<T, TI, EMBEDDING_OUTPUT_SHAPE>;
    using OUTPUT_SHAPE = rlt::tensor::Shape<TI, SEQUENCE_LENGTH, BATCH_SIZE, OUTPUT_DIM>;
    using OUTPUT_SPEC = rlt::tensor::Specification<T, TI, OUTPUT_SHAPE>;
    rlt::Tensor<INPUT_SPEC> input;
    rlt::Tensor<EMBEDDING_OUTPUT_SPEC> embedding_output;
    rlt::Tensor<OUTPUT_SPEC> output;
    rlt::malloc(device, embedding_layer);
    rlt::malloc(device, embedding_buffer);
    rlt::malloc(device, gru);
    rlt::malloc(device, gru_buffer);
    rlt::malloc(device, input);
    rlt::malloc(device, embedding_output);
    rlt::malloc(device, output);
    rlt::init_weights(device, embedding_layer, rng);
    rlt::init_weights(device, gru, rng);
    rlt::print(device, embedding_layer.weights.parameters);
    for(TI epoch_i=0; epoch_i < 10; epoch_i++){
        std::shuffle(dataset.begin(), dataset.end(), rng);
        for(TI sample_i=0; sample_i < dataset.size(); sample_i += BATCH_SIZE){
            for(TI batch_i = 0; batch_i < BATCH_SIZE; batch_i++){
                for(TI sequence_i = 0; sequence_i < SEQUENCE_LENGTH; sequence_i++){
                    rlt::set(device, input, std::get<0>(dataset[sample_i + batch_i])[sequence_i], sequence_i, batch_i);
                }
            }
            rlt::print(device, input);
            rlt::forward(device, embedding_layer, input, embedding_output, embedding_buffer, rng);
            rlt::evaluate(device, gru, embedding_output, output, gru_buffer, rng);
            rlt::print(device, output);
        }
    }

    return 0;
}
